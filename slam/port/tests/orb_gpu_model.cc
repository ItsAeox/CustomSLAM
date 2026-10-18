// GPU ORB model (2026-10-19, claude/PLAN.md 4c): ORB-SLAM3's monocular feature extraction (ORBextractor::operator())
// rewritten as the per-pixel integer steps a GPU can run, checked stage by stage against ORBextractor on the same
// images. Whatever matches here exactly is what the WGSL port has to reproduce.
//   pyramid  OpenCV's INTER_LINEAR resize (8-bit path) from its own coefficient tables: horizontal integer taps; vertical
//            in the SIMD form (VResizeLinearVec_32s8u: ((H0 >> 4) * b0 >> 16) + ((H1 >> 4) * b1 >> 16), rounded >> 2)
//            for every column: measured 19 Oct, native and WASM (the plain FixedPtCast tail read from the source
//            does not match: simdLanes=16 gives +1 in the last 5-7 columns of levels 2-7)
//   blur     OpenCV's bit-exact 8-bit Gaussian (7 taps, sigma 2, 8.8 fixed point, reflect-101 borders)
//   FAST     one score map per level (FAST-9 score = strongest 9-pixel arc contrast - 1); ORB-SLAM3's ~35-px cells with
//            threshold 20, or 7 where no corner survives at 20; strict 3x3 non-maximum suppression inside the cell;
//            candidates in ORB-SLAM3's order (cell by cell, then row by row)
//   octree   the unchanged ORBextractor::DistributeOctTree; then IC_Angle and rotated BRIEF as in ORBextractor.cc
// Usage: orb_gpu_model <image | directory of frame_*.jpg> [nFeatures=1000] [maxImages=20] [simdLanes=1] [levels=8]

#include "ORBextractor.h"
#include "../web/WebOrbGpu.h"
#include "../web/WebOrbGpuHost.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using std::vector;

namespace {

const int EDGE_THRESHOLD = webgpuorb::EDGE_THRESHOLD, PATCH_SIZE = 31, HALF_PATCH_SIZE = 15;

struct Probe : ORB_SLAM3::ORBextractor {  // ORBextractor's protected parts
  using ORBextractor::ORBextractor;
  using ORBextractor::DistributeOctTree;
  using ORBextractor::umax;
  using ORBextractor::pattern;
  using ORBextractor::mnFeaturesPerLevel;
  using ORBextractor::mvScaleFactor;
  using ORBextractor::mvInvScaleFactor;
  using ORBextractor::nlevels;
  using ORBextractor::iniThFAST;
  using ORBextractor::minThFAST;
};

// ---- pyramid: cv::resize(src, dst, dsz, 0, 0, INTER_LINEAR) for CV_8UC1 (hal::resize -> resizeGeneric_) ----
cv::Mat modelResize(const cv::Mat& src, cv::Size dsz, int simdLanes) {
  const int sw = src.cols, sh = src.rows, dw = dsz.width, dh = dsz.height;
  const double scale_x = 1. / ((double)dw / sw), scale_y = 1. / ((double)dh / sh);
  vector<int> xofs(dw), yofs(dh);
  vector<short> ia(2 * dw), ib(2 * dh);
  int xmax = dw;
  for (int dx = 0; dx < dw; dx++) {
    float fx = (float)((dx + 0.5) * scale_x - 0.5);
    int sx = cvFloor(fx);
    fx -= sx;
    if (sx < 0) fx = 0, sx = 0;
    if (sx + 1 >= sw) {
      xmax = std::min(xmax, dx);
      if (sx >= sw - 1) fx = 0, sx = sw - 1;
    }
    xofs[dx] = sx;
    ia[2 * dx] = cv::saturate_cast<short>((1.f - fx) * 2048);
    ia[2 * dx + 1] = cv::saturate_cast<short>(fx * 2048);
  }
  for (int dy = 0; dy < dh; dy++) {
    float fy = (float)((dy + 0.5) * scale_y - 0.5);
    int sy = cvFloor(fy);
    fy -= sy;
    yofs[dy] = sy;
    ib[2 * dy] = cv::saturate_cast<short>((1.f - fy) * 2048);
    ib[2 * dy + 1] = cv::saturate_cast<short>(fy * 2048);
  }
  // VResizeLinearVec_32s8u covers 16-column blocks, then one 8-column block if more than 8 are left
  const int simdEnd = simdLanes <= 0 ? 0 : (dw / simdLanes) * simdLanes + ((dw % simdLanes) > simdLanes / 2 ? simdLanes / 2 : 0);
  cv::Mat dst(dh, dw, CV_8U);
  vector<int> H0(dw), H1(dw);
  auto hrow = [&](int sy, vector<int>& H) {
    const uchar* S = src.ptr<uchar>(sy);
    for (int dx = 0; dx < dw; dx++)
      H[dx] = dx < xmax ? S[xofs[dx]] * ia[2 * dx] + S[xofs[dx] + 1] * ia[2 * dx + 1] : S[xofs[dx]] * 2048;
  };
  for (int dy = 0; dy < dh; dy++) {
    hrow(std::clamp(yofs[dy], 0, sh - 1), H0);
    hrow(std::clamp(yofs[dy] + 1, 0, sh - 1), H1);
    const int b0 = ib[2 * dy], b1 = ib[2 * dy + 1];
    uchar* D = dst.ptr<uchar>(dy);
    for (int x = 0; x < dw; x++) {
      int v;
      if (x < simdEnd)  // v_rshr_pack_u<2>(v_mul_hi(H0 >> 4, b0) + v_mul_hi(H1 >> 4, b1))
        v = ((((H0[x] >> 4) * b0) >> 16) + (((H1[x] >> 4) * b1) >> 16) + 2) >> 2;
      else  // FixedPtCast<int, uchar, 22>
        v = (H0[x] * b0 + H1[x] * b1 + (1 << 21)) >> 22;
      D[x] = (uchar)std::clamp(v, 0, 255);
    }
  }
  return dst;
}

// ---- blur: GaussianBlur(img, Size(7, 7), 2, 2, BORDER_REFLECT_101) on a non-submatrix CV_8U image (fixed point) ----
inline int refl101(int p, int n) { return p < 0 ? -p : p >= n ? 2 * n - p - 2 : p; }

cv::Mat modelBlur(const cv::Mat& img, const vector<int>& m) {
  const int w = img.cols, h = img.rows;
  vector<unsigned> H((size_t)w * h);
  for (int y = 0; y < h; y++) {
    const uchar* S = img.ptr<uchar>(y);
    for (int x = 0; x < w; x++) {
      unsigned s = 0;
      for (int i = 0; i < 7; i++) s += (unsigned)m[i] * S[refl101(x + i - 3, w)];
      H[(size_t)y * w + x] = s;
    }
  }
  cv::Mat out(h, w, CV_8U);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      unsigned long s = 0;
      for (int j = 0; j < 7; j++) s += (unsigned long)m[j] * H[(size_t)refl101(y + j - 3, h) * w + x];
      out.at<uchar>(y, x) = (uchar)std::min<unsigned long>((s + 32768) >> 16, 255);
    }
  return out;
}

// ---- FAST-9 score (OpenCV FAST_t / cornerScore<16>, SIMD builds): strongest 9-arc contrast - 1, 0 if not a corner at t ----
const int CIRC[16][2] = {{0, 3}, {1, 3}, {2, 2}, {3, 1}, {3, 0}, {3, -1}, {2, -2}, {1, -3},
                         {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}};
int fastScore(const cv::Mat& img, int x, int y, int t) {
  const int c = img.at<uchar>(y, x);
  int d[16];
  for (int k = 0; k < 16; k++) d[k] = img.at<uchar>(y + CIRC[k][1], x + CIRC[k][0]) - c;
  int best = 0;
  for (int k = 0; k < 16; k++) {
    int mn = 1000, mx = -1000;
    for (int j = 0; j < 9; j++) { mn = std::min(mn, d[(k + j) & 15]); mx = std::max(mx, d[(k + j) & 15]); }
    best = std::max(best, std::max(mn, -mx));
  }
  return best > t ? best - 1 : 0;
}

using webgpuorb::CellGrid;

// reference: ORB-SLAM3's cell loop with cv::FAST (what DistributeOctTree receives), on a given level image
vector<cv::KeyPoint> refCandidates(const cv::Mat& L, int iniTh, int minTh) {
  CellGrid g(L.cols, L.rows);
  vector<cv::KeyPoint> out;
  for (int i = 0; i < g.nRows; i++)
    for (int j = 0; j < g.nCols; j++) {
      int x0, y0, x1, y1;
      if (!g.roi(i, j, x0, y0, x1, y1)) continue;
      vector<cv::KeyPoint> cell;
      cv::FAST(L.rowRange(y0, y1).colRange(x0, x1), cell, iniTh, true);
      if (cell.empty()) cv::FAST(L.rowRange(y0, y1).colRange(x0, x1), cell, minTh, true);
      for (auto& k : cell) { k.pt.x += j * g.wCell; k.pt.y += i * g.hCell; out.push_back(k); }
    }
  return out;
}

// model: one score map at threshold 7, then per cell: threshold, strict NMS inside the cell's detection area, order
vector<cv::KeyPoint> modelCandidates(const cv::Mat& L, int iniTh, int minTh) {
  const int w = L.cols, h = L.rows;
  cv::Mat s(h, w, CV_32S, cv::Scalar(0));
  for (int y = 3; y < h - 3; y++)
    for (int x = 3; x < w - 3; x++) s.at<int>(y, x) = fastScore(L, x, y, minTh);
  CellGrid g(w, h);
  vector<cv::KeyPoint> out;
  for (int i = 0; i < g.nRows; i++)
    for (int j = 0; j < g.nCols; j++) {
      int x0, y0, x1, y1;
      if (!g.roi(i, j, x0, y0, x1, y1)) continue;
      const int ax0 = x0 + 3, ax1 = x1 - 3, ay0 = y0 + 3, ay1 = y1 - 3;  // FAST's detection area in the cell
      auto survives = [&](int x, int y, int t) {
        const int v = s.at<int>(y, x);
        if (v < t) return false;  // corner at t <=> score >= t
        for (int dy = -1; dy <= 1; dy++)
          for (int dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;
            const int xx = x + dx, yy = y + dy;
            if (xx < ax0 || xx >= ax1 || yy < ay0 || yy >= ay1) continue;  // outside the cell: 0
            if (s.at<int>(yy, xx) >= v) return false;  // strict maximum (ties suppress both)
          }
        return true;
      };
      int t = iniTh;
      bool any = false;
      for (int y = ay0; y < ay1 && !any; y++)
        for (int x = ax0; x < ax1 && !any; x++) any = survives(x, y, iniTh);
      if (!any) t = minTh;
      for (int y = ay0; y < ay1; y++)
        for (int x = ax0; x < ax1; x++)
          if (survives(x, y, t))
            out.emplace_back((float)(x - g.minBX), (float)(y - g.minBY), 7.f, -1.f, (float)s.at<int>(y, x));
    }
  return out;
}

// the GPU form of the same step (what the WGSL kernels do): per-pixel cell maps (WebOrbGpu.h), a per-cell flag
// "a pixel survives at 20", candidates emitted in any order, then sorted into ORB-SLAM3's order on the CPU
vector<cv::KeyPoint> gpuCandidates(const cv::Mat& L, int iniTh, int minTh) {
  const int w = L.cols, h = L.rows;
  CellGrid g(w, h);
  vector<int> colCell, rowCell;
  g.maps(w, h, colCell, rowCell);
  cv::Mat s(h, w, CV_32S, cv::Scalar(0));
  for (int y = 3; y < h - 3; y++)
    for (int x = 3; x < w - 3; x++) s.at<int>(y, x) = fastScore(L, x, y, minTh);
  auto strictMax = [&](int x, int y) {
    const int v = s.at<int>(y, x);
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++)
        if ((dx || dy) && colCell[x + dx] == colCell[x] && rowCell[y + dy] == rowCell[y] && s.at<int>(y + dy, x + dx) >= v)
          return false;
    return true;
  };
  vector<char> flag20(g.nCols * g.nRows, 0);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      if (colCell[x] >= 0 && rowCell[y] >= 0 && s.at<int>(y, x) >= iniTh && strictMax(x, y)) flag20[rowCell[y] * g.nCols + colCell[x]] = 1;
  struct C { int key, y, x, sc; };
  vector<C> cs;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      if (colCell[x] < 0 || rowCell[y] < 0) continue;
      const int cell = rowCell[y] * g.nCols + colCell[x], t = flag20[cell] ? iniTh : minTh, v = s.at<int>(y, x);
      if (v >= t && strictMax(x, y)) cs.push_back({cell, y, x, v});
    }
  std::sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.key != b.key ? a.key < b.key : a.y != b.y ? a.y < b.y : a.x < b.x; });
  vector<cv::KeyPoint> out;
  for (const C& c : cs) out.emplace_back((float)(c.x - g.minBX), (float)(c.y - g.minBY), 7.f, -1.f, (float)c.sc);
  return out;
}

// ---- orientation + rotated BRIEF (ORBextractor.cc IC_Angle / computeOrbDescriptor) ----
float icAngle(const cv::Mat& image, cv::Point2f pt, const vector<int>& u_max) {
  int m_01 = 0, m_10 = 0;
  const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));
  for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u) m_10 += u * center[u];
  const int step = (int)image.step1();
  for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
    int v_sum = 0;
    const int d = u_max[v];
    for (int u = -d; u <= d; ++u) {
      const int val_plus = center[u + v * step], val_minus = center[u - v * step];
      v_sum += (val_plus - val_minus);
      m_10 += u * (val_plus + val_minus);
    }
    m_01 += v * v_sum;
  }
  return cv::fastAtan2((float)m_01, (float)m_10);
}

void orbDescriptor(const cv::KeyPoint& kpt, const cv::Mat& img, const cv::Point* pattern, uchar* desc) {
  const float angle = (float)kpt.angle * (float)(CV_PI / 180.f);
  const float a = (float)cos(angle), b = (float)sin(angle);
  const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
  const int step = (int)img.step;
  auto val = [&](int idx) { return center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + cvRound(pattern[idx].x * a - pattern[idx].y * b)]; };
  for (int i = 0; i < 32; ++i, pattern += 16) {
    int v = 0;
    for (int bit = 0; bit < 8; bit++) v |= (val(2 * bit) < val(2 * bit + 1)) << bit;
    desc[i] = (uchar)v;
  }
}

// what the GPU computes for the host path: IC_Angle's integer moments, and rotated BRIEF from a given cos / sin
void icMoments(const cv::Mat& image, int x, int y, const vector<int>& u_max, int& m_01, int& m_10) {
  m_01 = 0; m_10 = 0;
  const uchar* center = &image.at<uchar>(y, x);
  for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u) m_10 += u * center[u];
  const int step = (int)image.step1();
  for (int v = 1; v <= HALF_PATCH_SIZE; ++v) {
    int v_sum = 0;
    const int d = u_max[v];
    for (int u = -d; u <= d; ++u) {
      const int val_plus = center[u + v * step], val_minus = center[u - v * step];
      v_sum += (val_plus - val_minus);
      m_10 += u * (val_plus + val_minus);
    }
    m_01 += v * v_sum;
  }
}

void descFromCosSin(const cv::Mat& img, int x, int y, float a, float b, const cv::Point* pattern, uchar* desc) {
  const uchar* center = &img.at<uchar>(y, x);
  const int step = (int)img.step;
  auto val = [&](int idx) { return center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + cvRound(pattern[idx].x * a - pattern[idx].y * b)]; };
  for (int i = 0; i < 32; ++i, pattern += 16) {
    int v = 0;
    for (int bit = 0; bit < 8; bit++) v |= (val(2 * bit) < val(2 * bit + 1)) << bit;
    desc[i] = (uchar)v;
  }
}

long mismatch(const cv::Mat& a, const cv::Mat& b) {
  if (a.size() != b.size()) return -1;
  cv::Mat d;
  cv::compare(a, b, d, cv::CMP_NE);
  return cv::countNonZero(d);
}

bool sameKp(const cv::KeyPoint& a, const cv::KeyPoint& b) {
  return a.pt == b.pt && a.response == b.response && a.octave == b.octave && a.angle == b.angle && a.size == b.size;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: orb_gpu_model <image|dir> [nFeatures=1000] [maxImages=20] [simdLanes=1] [levels=8]\n"); return 2; }
  const int nFeatures = argc > 2 ? std::stoi(argv[2]) : 1000, maxImages = argc > 3 ? std::stoi(argv[3]) : 20;
  const int simdLanes = argc > 4 ? std::stoi(argv[4]) : 1;  // 1 = every column in the SIMD form
  const int nLevels = argc > 5 ? std::stoi(argv[5]) : 8;
  vector<std::string> files;
  if (fs::is_directory(argv[1])) {
    vector<std::pair<int, std::string>> v;
    for (auto& e : fs::directory_iterator(argv[1])) {
      const std::string n = e.path().filename().string();
      if (n.rfind("frame_", 0) == 0) v.emplace_back(std::stoi(n.substr(6)), e.path().string());
    }
    std::sort(v.begin(), v.end());
    const int stride = std::max<int>(1, (int)v.size() / std::max(1, maxImages));
    for (size_t i = 0; i < v.size() && (int)files.size() < maxImages; i += stride) files.push_back(v[i].second);
  } else files.push_back(argv[1]);

  const vector<int> taps = webgpuorb::gaussTaps();
  std::printf("blur taps (8.8 fixed point): %d %d %d %d %d %d %d; resize SIMD lanes %d\n", taps[0], taps[1], taps[2], taps[3],
              taps[4], taps[5], taps[6], simdLanes);
  long totPyr = 0, totBlur = 0, totCandDiff = 0, totKpDiff = 0, totDescDiff = 0, totKp = 0, imgsExact = 0, totHostKp = 0, totHostDesc = 0;
  for (const auto& f : files) {
    cv::Mat im = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if (im.empty()) { std::printf("%s: unreadable\n", f.c_str()); continue; }
    Probe ref(nFeatures, 1.2f, nLevels, 20, 7);
    vector<cv::KeyPoint> rk;
    cv::Mat rd;
    vector<int> lap = {0, 1000};  // Frame.cc monocular: ExtractORB(0, imGray, 0, 1000)
    ref(im, cv::Mat(), rk, rd, lap);

    // model pyramid (from the model's own previous level, as ComputePyramid)
    vector<cv::Mat> pyr(ref.nlevels);
    pyr[0] = im;
    long pyrDiff = 0, blurDiff = 0, candDiff = 0;
    std::string lv;
    vector<vector<cv::KeyPoint>> levelKps(ref.nlevels);
    vector<cv::Mat> levelDesc(ref.nlevels), blurs(ref.nlevels);
    for (int l = 0; l < ref.nlevels; l++) {
      if (l) {
        const float s = ref.mvInvScaleFactor[l];
        const cv::Size sz(cvRound((float)im.cols * s), cvRound((float)im.rows * s));
        pyr[l] = simdLanes == 1 ? webgpuorb::applyResize(pyr[l - 1], webgpuorb::resizeTables(pyr[l - 1].cols, pyr[l - 1].rows, sz.width, sz.height))
                                : modelResize(pyr[l - 1], sz, simdLanes);
      }
      const long pd = mismatch(pyr[l], ref.mvImagePyramid[l]);
      if (pd > 0 && getenv("ORB_MODEL_WHERE")) {  // diagnostic: which columns / rows differ, first few values
        vector<int> cols(pyr[l].cols, 0), rows(pyr[l].rows, 0);
        std::string ex;
        for (int y = 0; y < pyr[l].rows; y++)
          for (int x = 0; x < pyr[l].cols; x++) {
            const int a = pyr[l].at<uchar>(y, x), b = ref.mvImagePyramid[l].at<uchar>(y, x);
            if (a != b) {
              cols[x]++; rows[y]++;
              if (ex.size() < 300) ex += " (" + std::to_string(x) + "," + std::to_string(y) + ":" + std::to_string(a) + "/" + std::to_string(b) + ")";
            }
          }
        std::string cs;
        for (int x = 0; x < pyr[l].cols; x++) if (cols[x]) cs += " " + std::to_string(x) + ":" + std::to_string(cols[x]);
        int nr = 0;
        for (int y = 0; y < pyr[l].rows; y++) nr += rows[y] > 0;
        std::printf("  L%d %dx%d differ: columns%s; rows with a difference %d; e.g.%s\n", l, pyr[l].cols, pyr[l].rows, cs.c_str(), nr, ex.c_str());
      }
      cv::Mat rb = ref.mvImagePyramid[l].clone(), mb = modelBlur(pyr[l], taps);
      blurs[l] = mb;
      cv::GaussianBlur(rb, rb, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
      const long bd = mismatch(mb, rb);
      // candidates: reference = ORB-SLAM3's cell loop with cv::FAST on the reference level image
      const vector<cv::KeyPoint> rc = refCandidates(ref.mvImagePyramid[l], ref.iniThFAST, ref.minThFAST);
      const vector<cv::KeyPoint> mc0 = modelCandidates(pyr[l], ref.iniThFAST, ref.minThFAST);
      const vector<cv::KeyPoint> mc = gpuCandidates(pyr[l], ref.iniThFAST, ref.minThFAST);  // the GPU form feeds the octree
      long cd = (long)std::abs((long)rc.size() - (long)mc.size()) + (long)std::abs((long)mc0.size() - (long)mc.size());
      for (size_t k = 0; k < std::min(rc.size(), mc.size()); k++) cd += !sameKp(rc[k], mc[k]);
      for (size_t k = 0; k < std::min(mc0.size(), mc.size()); k++) cd += !sameKp(mc0[k], mc[k]);
      pyrDiff += pd; blurDiff += bd; candDiff += cd;
      char b[96];
      std::snprintf(b, sizeof b, " L%d:%ld/%ld/%zu-%zu", l, pd, bd, rc.size(), mc.size());
      lv += b;
      // octree, orientation, descriptors on the model's images (ORBextractor::ComputeKeyPointsOctTree + operator())
      CellGrid g(pyr[l].cols, pyr[l].rows);
      vector<cv::KeyPoint> kps = ref.DistributeOctTree(mc, g.minBX, g.maxBX, g.minBY, g.maxBY, ref.mnFeaturesPerLevel[l], l);
      const int scaledPatchSize = PATCH_SIZE * ref.mvScaleFactor[l];
      for (auto& k : kps) { k.pt.x += g.minBX; k.pt.y += g.minBY; k.octave = l; k.size = scaledPatchSize; }
      for (auto& k : kps) k.angle = icAngle(pyr[l], k.pt, ref.umax);
      levelDesc[l] = cv::Mat((int)kps.size(), 32, CV_8U);
      for (size_t k = 0; k < kps.size(); k++) orbDescriptor(kps[k], mb, &ref.pattern[0], levelDesc[l].ptr((int)k));
      levelKps[l] = kps;
    }
    // output order as ORBextractor::operator() with the monocular lapping area [0, 1000]
    int n = 0;
    for (auto& v : levelKps) n += (int)v.size();
    vector<cv::KeyPoint> mk(n);
    cv::Mat md(n, 32, CV_8U);
    int mono = 0, stereo = n - 1;
    for (int l = 0; l < ref.nlevels; l++)
      for (size_t k = 0; k < levelKps[l].size(); k++) {
        cv::KeyPoint kp = levelKps[l][k];
        if (l) kp.pt *= ref.mvScaleFactor[l];
        const int at = (kp.pt.x >= lap[0] && kp.pt.x <= lap[1]) ? stereo-- : mono++;
        mk[at] = kp;
        levelDesc[l].row((int)k).copyTo(md.row(at));
      }
    long kd = (long)std::abs((long)rk.size() - (long)mk.size()), dd = 0;
    for (size_t k = 0; k < std::min(rk.size(), mk.size()); k++) {
      kd += !sameKp(rk[k], mk[k]);
      dd += std::memcmp(rd.ptr((int)k), md.ptr((int)k), 32) != 0;
    }
    totPyr += pyrDiff; totBlur += blurDiff; totCandDiff += candDiff; totKpDiff += kd; totDescDiff += dd; totKp += (long)rk.size();
    imgsExact += (pyrDiff == 0 && blurDiff == 0 && candDiff == 0 && kd == 0 && dd == 0);
    // host path (WebOrbGpuHost.h, what the browser runs) with the model playing the GPU: candidate words + moments in a
    // shuffled order (the GPU's atomics give any order), then rotated BRIEF from the host's pass-2 cos / sin
    webgpuorb::OrbGpuHost host(nFeatures, 1.2f, nLevels, 20, 7, im.cols, im.rows);
    im.copyTo(host.frame());
    host.reference();
    vector<uint32_t> cand;
    for (int l = 0; l < ref.nlevels; l++) {
      CellGrid g(pyr[l].cols, pyr[l].rows);
      for (const auto& k : gpuCandidates(pyr[l], ref.iniThFAST, ref.minThFAST)) {
        const int x = (int)k.pt.x + g.minBX, y = (int)k.pt.y + g.minBY;
        int m01, m10;
        icMoments(pyr[l], x, y, ref.umax, m01, m10);
        cand.insert(cand.end(), {(uint32_t)x | ((uint32_t)y << 16), (uint32_t)l | ((uint32_t)k.response << 8), (uint32_t)m01, (uint32_t)m10});
      }
    }
    const int nc = (int)cand.size() / 4;
    for (int i = nc - 1; i > 0; i--) {  // deterministic shuffle of the 4-word groups
      const int j = (int)(((unsigned)i * 2654435761u) % (unsigned)(i + 1));
      for (int w = 0; w < 4; w++) std::swap(cand[4 * i + w], cand[4 * j + w]);
    }
    const int kept = std::max(0, host.distribute(cand.data(), nc));  // -1 = more kept keypoints than capKp
    vector<uint32_t> gdesc((size_t)kept * 8);
    for (int i = 0; i < kept; i++) {
      const uint32_t* q = &host.pass2_[(size_t)i * 8];
      float a, b;
      std::memcpy(&a, &q[3], 4); std::memcpy(&b, &q[4], 4);
      descFromCosSin(blurs[q[2]], (int)q[0], (int)q[1], a, b, &ref.pattern[0], reinterpret_cast<uchar*>(&gdesc[(size_t)i * 8]));
    }
    const vector<int> hr = host.compare(gdesc.data(), kept);
    totHostKp += hr[0]; totHostDesc += hr[1];
    std::printf("%s: pyramid px %ld, blur px %ld, candidates %ld, keypoints %zu/%zu differ %ld, descriptors differ %ld; host path: %d candidates, keypoints differ %d, descriptors %d (bits %d) |%s\n",
                fs::path(f).filename().string().c_str(), pyrDiff, blurDiff, candDiff, rk.size(), mk.size(), kd, dd, nc, hr[0], hr[1], hr[2], lv.c_str());
  }
  std::printf("TOTAL %zu images, %ld exact: pyramid px %ld, blur px %ld, candidates %ld, keypoints %ld of %ld, descriptors %ld; host path: keypoints %ld, descriptors %ld\n",
              files.size(), imgsExact, totPyr, totBlur, totCandDiff, totKpDiff, totKp, totDescDiff, totHostKp, totHostDesc);
  return 0;
}
