// GPU ORB host side (claude/PLAN.md 4c; GPU side web/gpuorb.js): the constants the GPU needs (WebOrbGpu.h tables), the
// reference extraction (ORBextractor), and the CPU part of the GPU path: GPU candidates -> ORB-SLAM3's order -> the
// unchanged DistributeOctTree -> angles from the GPU's integer moments -> keypoints in ORBextractor::operator()'s output
// order (monocular lapping area [0, 1000]) + the pass-2 input (level x, y, level, cos, sin as computeOrbDescriptor).
// Plain C++: the browser binding (bindings.cc OrbGpuHost) and slam/port/tests/orb_gpu_model.cc use it.
#pragma once

#include "ORBextractor.h"
#include "WebOrbGpu.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace webgpuorb {

class OrbGpuHost {
  struct Ext : ORB_SLAM3::ORBextractor {  // ORBextractor's protected parts
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

 public:
  // candidates do not depend on nFeatures (same FAST thresholds); kept keypoints do (the initialiser extracts 5x)
  static const int CAP_CAND = 32768, PATCH_SIZE = 31;
  const int capKp;  // kept-keypoint capacity: the octree keeps at most a little over nFeatures

  OrbGpuHost(int nFeatures, float scaleFactor, int nLevels, int iniTh, int minTh, int cols, int rows)
      : capKp(2 * nFeatures + 256), ext_(nFeatures, scaleFactor, nLevels, iniTh, minTh), cols_(cols), rows_(rows) {
    frame_.create(rows, cols, CV_8U);
    const int L = ext_.nlevels;
    sizes_.resize(L);
    for (int l = 0; l < L; l++) {  // ORBextractor::ComputePyramid's level sizes
      const float s = ext_.mvInvScaleFactor[l];
      sizes_[l] = cv::Size(cvRound((float)cols * s), cvRound((float)rows * s));
    }
    // constants (layout: web/gpuorb.js)
    consts_.assign(8 + 16 * L, 0);
    int pixels = 0, cells = 0;
    std::vector<int> data;
    auto put = [&](const std::vector<int>& v) { const int o = (int)(consts_.size() + data.size()); data.insert(data.end(), v.begin(), v.end()); return o; };
    grids_.clear();
    colCell_.resize(L); rowCell_.resize(L);
    for (int l = 0; l < L; l++) {
      int* h = &consts_[8 + 16 * l];
      const int w = sizes_[l].width, hh = sizes_[l].height;
      h[0] = pixels; h[1] = w; h[2] = hh;
      pixels += w * hh;
      if (l) {
        const ResizeTables t = resizeTables(sizes_[l - 1].width, sizes_[l - 1].height, w, hh);
        std::vector<int> a0(w), a1(w), b0(hh), b1(hh);
        for (int x = 0; x < w; x++) { a0[x] = t.ia[2 * x]; a1[x] = t.ia[2 * x + 1]; }
        for (int y = 0; y < hh; y++) { b0[y] = t.ib[2 * y]; b1[y] = t.ib[2 * y + 1]; }
        h[3] = put(t.xofs); h[4] = put(a0); h[5] = put(a1);
        h[6] = put(t.yofs); h[7] = put(b0); h[8] = put(b1);
        h[13] = t.xmax;
      }
      grids_.emplace_back(w, hh);
      grids_[l].maps(w, hh, colCell_[l], rowCell_[l]);
      h[9] = put(colCell_[l]); h[10] = put(rowCell_[l]);
      h[11] = grids_[l].nCols; h[12] = cells;
      cells += grids_[l].nCols * grids_[l].nRows;
    }
    std::vector<int> umax(ext_.umax.begin(), ext_.umax.end()), pat;
    for (const cv::Point& p : ext_.pattern) { pat.push_back(p.x); pat.push_back(p.y); }
    consts_[0] = L; consts_[1] = pixels;
    consts_[2] = put(umax); consts_[3] = put(pat); consts_[4] = put(gaussTaps());
    consts_[5] = CAP_CAND; consts_[6] = capKp; consts_[7] = cells;
    consts_.insert(consts_.end(), data.begin(), data.end());
  }

  const std::vector<int32_t>& consts() const { return consts_; }
  cv::Mat& frame() { return frame_; }
  int levels() const { return ext_.nlevels; }

  // ORBextractor on frame() (what Frame.cc's monocular constructor calls)
  int reference() {
    std::vector<int> lap = {0, 1000};
    ext_(frame_, cv::Mat(), refKps_, refDesc_, lap);
    return (int)refKps_.size();
  }

  // GPU pass-1 output: n candidates, 4 words each [x | y << 16, level | score << 8, m01, m10] (all of them: when the
  // GPU had more than CAP_CAND the caller must use CPU ORB instead). Returns the number of keypoints kept, or -1 if
  // more than capKp (then also CPU ORB); fills kps_ (final keypoints, output order) and pass2_ (8 words each: x, y,
  // level, cos, sin bits).
  int distribute(const uint32_t* cand, int n) {
    const int L = ext_.nlevels;
    struct C { int key, y, x, sc, m01, m10; };
    std::vector<std::vector<C>> per(L);
    for (int i = 0; i < n; i++) {
      const uint32_t p = cand[4 * i], q = cand[4 * i + 1];
      const int x = (int)(p & 0xFFFF), y = (int)(p >> 16), l = (int)(q & 255), sc = (int)(q >> 8);
      if (l >= L) continue;
      per[l].push_back({rowCell_[l][y] * grids_[l].nCols + colCell_[l][x], y, x, sc, (int)cand[4 * i + 2], (int)cand[4 * i + 3]});
    }
    std::vector<std::vector<cv::KeyPoint>> levelKps(L);
    std::vector<std::vector<int>> levelMom(L);
    for (int l = 0; l < L; l++) {
      std::vector<C>& v = per[l];
      std::sort(v.begin(), v.end(), [](const C& a, const C& b) { return a.key != b.key ? a.key < b.key : a.y != b.y ? a.y < b.y : a.x < b.x; });
      const CellGrid& g = grids_[l];
      std::vector<cv::KeyPoint> in;
      in.reserve(v.size());
      for (size_t k = 0; k < v.size(); k++)  // as cv::FAST's output after ORB-SLAM3's cell offset; class_id = index
        in.emplace_back((float)(v[k].x - g.minBX), (float)(v[k].y - g.minBY), 7.f, -1.f, (float)v[k].sc, 0, (int)k);
      std::vector<cv::KeyPoint> kps = ext_.DistributeOctTree(in, g.minBX, g.maxBX, g.minBY, g.maxBY, ext_.mnFeaturesPerLevel[l], l);
      const int scaledPatchSize = PATCH_SIZE * ext_.mvScaleFactor[l];
      for (cv::KeyPoint& k : kps) {
        const C& c = v[k.class_id];
        k.pt.x += g.minBX; k.pt.y += g.minBY; k.octave = l; k.size = (float)scaledPatchSize;
        k.angle = cv::fastAtan2((float)c.m01, (float)c.m10);  // IC_Angle
        k.class_id = -1;
      }
      levelKps[l] = kps;
    }
    int total = 0;
    for (auto& v : levelKps) total += (int)v.size();
    if (total > capKp) return -1;
    kps_.assign(total, cv::KeyPoint());
    pass2_.assign((size_t)total * 8, 0);
    int mono = 0, stereo = total - 1;
    const float factorPI = (float)(CV_PI / 180.f);
    for (int l = 0; l < L; l++)
      for (const cv::KeyPoint& k0 : levelKps[l]) {
        if (mono > stereo) break;
        cv::KeyPoint kp = k0;
        const float angle = (float)kp.angle * factorPI;  // computeOrbDescriptor
        const float a = (float)std::cos(angle), b = (float)std::sin(angle);
        const int cx = cvRound(kp.pt.x), cy = cvRound(kp.pt.y);
        if (l) kp.pt *= ext_.mvScaleFactor[l];
        const int at = (kp.pt.x >= 0 && kp.pt.x <= 1000) ? stereo-- : mono++;
        kps_[at] = kp;
        uint32_t* q = &pass2_[(size_t)at * 8];
        q[0] = (uint32_t)cx; q[1] = (uint32_t)cy; q[2] = (uint32_t)l;
        std::memcpy(&q[3], &a, 4); std::memcpy(&q[4], &b, 4);
      }
    return total;
  }

  // GPU descriptors (8 words per kept keypoint, output order) against the reference: [0] keypoints differing (or count
  // mismatch), [1] descriptors differing, [2] bits differing, [3] reference count, [4] GPU-path count
  std::vector<int> compare(const uint32_t* desc, int k) const {
    std::vector<int> r(5, 0);
    r[3] = (int)refKps_.size(); r[4] = k;
    r[0] = std::abs((int)refKps_.size() - k);
    for (int i = 0; i < std::min(k, (int)refKps_.size()); i++) {
      const cv::KeyPoint &a = refKps_[i], &b = kps_[i];
      r[0] += !(a.pt == b.pt && a.response == b.response && a.octave == b.octave && a.angle == b.angle && a.size == b.size);
      const uint8_t* g = reinterpret_cast<const uint8_t*>(desc + 8 * i);
      const uint8_t* c = refDesc_.ptr<uint8_t>(i);
      int bits = 0;
      for (int j = 0; j < 32; j++) bits += __builtin_popcount((unsigned)(g[j] ^ c[j]));
      r[1] += bits > 0; r[2] += bits;
    }
    return r;
  }

  std::vector<cv::KeyPoint> kps_, refKps_;
  std::vector<uint32_t> pass2_;
  cv::Mat refDesc_;

 private:
  Ext ext_;
  int cols_, rows_;
  cv::Mat frame_;
  std::vector<cv::Size> sizes_;
  std::vector<CellGrid> grids_;
  std::vector<std::vector<int>> colCell_, rowCell_;
  std::vector<int32_t> consts_;
};

}  // namespace webgpuorb
