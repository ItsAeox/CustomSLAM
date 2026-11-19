// GPU ORB (claude/PLAN.md 4c): the tables a GPU needs to compute ORB-SLAM3's monocular features bit-exactly with
// integer steps, built on the CPU with the same arithmetic as OpenCV / ORBextractor. Verified against ORBextractor by
// slam/port/tests/orb_gpu_model.cc (90/90 phone frames exact, WASM and native, 19 Oct). Per level:
//   resize  cv::resize(prev level, INTER_LINEAR) for 8-bit: column taps (xofs, ia0/ia1, xmax) and row taps (yofs,
//           ib0/ib1); H = S[sx] ia0 + S[sx+1] ia1 (S[sx] 2048 from xmax on), rows clamped to the image;
//           out = clamp((((H0 >> 4) ib0 >> 16) + ((H1 >> 4) ib1 >> 16) + 2) >> 2, 0, 255) for every column
//   blur    GaussianBlur 7x7, sigma 2, reflect-101, OpenCV's bit-exact fixed point: taps 18 34 48 56 48 34 18;
//           H = sum taps * pixel (16 bit), out = (sum taps * H + 32768) >> 16
//   FAST    score = strongest 9-arc contrast - 1 (0 if not a corner at 7); ORB-SLAM3's ~35-px cells: a pixel's cell
//           from colCell / rowCell (-1 = in no cell's detection area, the areas tile exactly); threshold 20, or 7 if no
//           pixel of the cell survives at 20; strict 3x3 maximum among neighbours of the same cell; the octree input
//           order is cell by cell (row-major), then row by row, then by column
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace webgpuorb {

const int EDGE_THRESHOLD = 19;

struct ResizeTables {
  int sw = 0, sh = 0, dw = 0, dh = 0, xmax = 0;
  std::vector<int> xofs, yofs;
  std::vector<short> ia, ib;  // 2 per column / row
};

// hal::resize's coefficient set-up for INTER_LINEAR, CV_8U (fixpt), source sw x sh to dw x dh
inline ResizeTables resizeTables(int sw, int sh, int dw, int dh) {
  ResizeTables t;
  t.sw = sw; t.sh = sh; t.dw = dw; t.dh = dh; t.xmax = dw;
  const double scale_x = 1. / ((double)dw / sw), scale_y = 1. / ((double)dh / sh);
  t.xofs.resize(dw); t.yofs.resize(dh); t.ia.resize(2 * dw); t.ib.resize(2 * dh);
  for (int dx = 0; dx < dw; dx++) {
    float fx = (float)((dx + 0.5) * scale_x - 0.5);
    int sx = cvFloor(fx);
    fx -= sx;
    if (sx < 0) fx = 0, sx = 0;
    if (sx + 1 >= sw) {
      t.xmax = std::min(t.xmax, dx);
      if (sx >= sw - 1) fx = 0, sx = sw - 1;
    }
    t.xofs[dx] = sx;
    t.ia[2 * dx] = cv::saturate_cast<short>((1.f - fx) * 2048);
    t.ia[2 * dx + 1] = cv::saturate_cast<short>(fx * 2048);
  }
  for (int dy = 0; dy < dh; dy++) {
    float fy = (float)((dy + 0.5) * scale_y - 0.5);
    int sy = cvFloor(fy);
    fy -= sy;
    t.yofs[dy] = sy;
    t.ib[2 * dy] = cv::saturate_cast<short>((1.f - fy) * 2048);
    t.ib[2 * dy + 1] = cv::saturate_cast<short>(fy * 2048);
  }
  return t;
}

// the resize itself (CPU form of the GPU kernel)
inline cv::Mat applyResize(const cv::Mat& src, const ResizeTables& t) {
  cv::Mat dst(t.dh, t.dw, CV_8U);
  std::vector<int> H0(t.dw), H1(t.dw);
  auto hrow = [&](int sy, std::vector<int>& H) {
    const uchar* S = src.ptr<uchar>(sy);
    for (int dx = 0; dx < t.dw; dx++)
      H[dx] = dx < t.xmax ? S[t.xofs[dx]] * t.ia[2 * dx] + S[t.xofs[dx] + 1] * t.ia[2 * dx + 1] : S[t.xofs[dx]] * 2048;
  };
  for (int dy = 0; dy < t.dh; dy++) {
    hrow(std::clamp(t.yofs[dy], 0, t.sh - 1), H0);
    hrow(std::clamp(t.yofs[dy] + 1, 0, t.sh - 1), H1);
    const int b0 = t.ib[2 * dy], b1 = t.ib[2 * dy + 1];
    uchar* D = dst.ptr<uchar>(dy);
    for (int x = 0; x < t.dw; x++)
      D[x] = (uchar)std::clamp(((((H0[x] >> 4) * b0) >> 16) + (((H1[x] >> 4) * b1) >> 16) + 2) >> 2, 0, 255);
  }
  return dst;
}

// getGaussianKernelFixedPoint_ED(getGaussianKernelBitExact(7, 2), 8 fraction bits): 18 34 48 56 48 34 18
inline std::vector<int> gaussTaps() {
  cv::Mat k = cv::getGaussianKernel(7, 2, CV_64F);
  std::vector<int> r(7);
  double err = 0;
  long sum = 0;
  for (int i = 0; i < 3; i++) {
    const double adj = k.at<double>(i) * 256 + err;
    const long v0 = std::lrint(adj);
    err = adj - (double)v0;
    r[i] = r[6 - i] = (int)v0;
    sum += v0;
  }
  r[3] = (int)(256 - 2 * sum);
  return r;
}

// ORBextractor::ComputeKeyPointsOctTree's cells for one level image (cols x rows)
struct CellGrid {
  int minBX, minBY, maxBX, maxBY, nCols, nRows, wCell, hCell;
  CellGrid(int cols, int rows) {
    minBX = EDGE_THRESHOLD - 3; minBY = minBX;
    maxBX = cols - EDGE_THRESHOLD + 3; maxBY = rows - EDGE_THRESHOLD + 3;
    const float W = 35, width = (float)(maxBX - minBX), height = (float)(maxBY - minBY);
    nCols = (int)(width / W); nRows = (int)(height / W);
    wCell = (int)std::ceil(width / nCols); hCell = (int)std::ceil(height / nRows);
  }
  // cell (i, j) ROI: rows [y0, y1), cols [x0, x1); false if ORB-SLAM3 skips it
  bool roi(int i, int j, int& x0, int& y0, int& x1, int& y1) const {
    const float iniY = (float)(minBY + i * hCell), iniX = (float)(minBX + j * wCell);
    float maxY = iniY + hCell + 6, maxX = iniX + wCell + 6;
    if (iniY >= maxBY - 3 || iniX >= maxBX - 6) return false;
    if (maxY > maxBY) maxY = (float)maxBY;
    if (maxX > maxBX) maxX = (float)maxBX;
    x0 = (int)iniX; y0 = (int)iniY; x1 = (int)maxX; y1 = (int)maxY;
    return true;
  }
  // per column / row: the cell whose FAST detection area (ROI minus its 3-pixel border) holds it, -1 = none
  void maps(int cols, int rows, std::vector<int>& colCell, std::vector<int>& rowCell) const {
    colCell.assign(cols, -1); rowCell.assign(rows, -1);
    for (int i = 0; i < nRows; i++)
      for (int j = 0; j < nCols; j++) {
        int x0, y0, x1, y1;
        if (!roi(i, j, x0, y0, x1, y1)) continue;
        for (int x = x0 + 3; x < x1 - 3; x++) colCell[x] = j;
        for (int y = y0 + 3; y < y1 - 3; y++) rowCell[y] = i;
      }
  }
};

}  // namespace webgpuorb
