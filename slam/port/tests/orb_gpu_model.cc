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
// Usage: orb_gpu_model <image | directory of frame_*.jpg> [nFeatures=1000] [maxImages=20] [simdLanes=1]

#include "ORBextractor.h"

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

const int EDGE_THRESHOLD = 19, PATCH_SIZE = 31, HALF_PATCH_SIZE = 15;

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
vector<int> gaussTaps() {  // getGaussianKernelFixedPoint_ED(getGaussianKernelBitExact(7, 2), 8 fraction bits)
  cv::Mat k = cv::getGaussianKernel(7, 2, CV_64F);
  vector<int> r(7);
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

struct CellGrid {  // ORBextractor::ComputeKeyPointsOctTree's cells for one level
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
};

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
  if (argc < 2) { std::fprintf(stderr, "usage: orb_gpu_model <image|dir> [nFeatures=1000] [maxImages=20] [simdLanes=1]\n"); return 2; }
  const int nFeatures = argc > 2 ? std::stoi(argv[2]) : 1000, maxImages = argc > 3 ? std::stoi(argv[3]) : 20;
  const int simdLanes = argc > 4 ? std::stoi(argv[4]) : 1;  // 1 = every column in the SIMD form
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

  const vector<int> taps = gaussTaps();
  std::printf("blur taps (8.8 fixed point): %d %d %d %d %d %d %d; resize SIMD lanes %d\n", taps[0], taps[1], taps[2], taps[3],
              taps[4], taps[5], taps[6], simdLanes);
  long totPyr = 0, totBlur = 0, totCandDiff = 0, totKpDiff = 0, totDescDiff = 0, totKp = 0, imgsExact = 0;
  for (const auto& f : files) {
    cv::Mat im = cv::imread(f, cv::IMREAD_GRAYSCALE);
    if (im.empty()) { std::printf("%s: unreadable\n", f.c_str()); continue; }
    Probe ref(nFeatures, 1.2f, 8, 20, 7);
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
    vector<cv::Mat> levelDesc(ref.nlevels);
    for (int l = 0; l < ref.nlevels; l++) {
      if (l) {
        const float s = ref.mvInvScaleFactor[l];
        pyr[l] = modelResize(pyr[l - 1], cv::Size(cvRound((float)im.cols * s), cvRound((float)im.rows * s)), simdLanes);
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
      cv::GaussianBlur(rb, rb, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
      const long bd = mismatch(mb, rb);
      // candidates: reference = ORB-SLAM3's cell loop with cv::FAST on the reference level image
      const vector<cv::KeyPoint> rc = refCandidates(ref.mvImagePyramid[l], ref.iniThFAST, ref.minThFAST);
      const vector<cv::KeyPoint> mc = modelCandidates(pyr[l], ref.iniThFAST, ref.minThFAST);
      long cd = (long)std::abs((long)rc.size() - (long)mc.size());
      for (size_t k = 0; k < std::min(rc.size(), mc.size()); k++) cd += !sameKp(rc[k], mc[k]);
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
    std::printf("%s: pyramid px %ld, blur px %ld, candidates %ld, keypoints %zu/%zu differ %ld, descriptors differ %ld |%s\n",
                fs::path(f).filename().string().c_str(), pyrDiff, blurDiff, candDiff, rk.size(), mk.size(), kd, dd, lv.c_str());
  }
  std::printf("TOTAL %zu images, %ld exact: pyramid px %ld, blur px %ld, candidates %ld, keypoints %ld of %ld, descriptors %ld\n",
              files.size(), imgsExact, totPyr, totBlur, totCandDiff, totKpDiff, totKp, totDescDiff);
  return 0;
}
