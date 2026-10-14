// Diagnostic: run ORB-SLAM3's feature extraction on one image and print checksums of every stage,
// so native and WebAssembly builds can be compared stage by stage.
//
// Usage: orb_dump <image> [clahe=1] [nFeatures=1500]

#include "ORBextractor.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ULL) {
  for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
  return h;
}

static uint64_t matHash(const cv::Mat& m) {
  cv::Mat c = m.isContinuous() ? m : m.clone();
  return fnv1a(c.data, c.total() * c.elemSize());
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: orb_dump <image> [clahe=1] [nFeatures=1500]\n"); return 2; }
  const bool clahe = argc < 3 || std::stoi(argv[2]) != 0;
  const int nFeatures = argc < 4 ? 1500 : std::stoi(argv[3]);

  cv::Mat im = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  std::printf("decoded      %dx%d hash=%016llx\n", im.cols, im.rows, (unsigned long long)matHash(im));
  if (clahe) {
    cv::Ptr<cv::CLAHE> op = cv::createCLAHE(3.0, cv::Size(8, 8));
    op->apply(im, im);
    std::printf("clahe        hash=%016llx\n", (unsigned long long)matHash(im));
  }

  // individual OpenCV primitives that ORBextractor relies on
  cv::Mat rs; cv::resize(im, rs, cv::Size(), 1 / 1.2, 1 / 1.2, cv::INTER_LINEAR);
  std::printf("resize       hash=%016llx\n", (unsigned long long)matHash(rs));
  cv::Mat bl; cv::GaussianBlur(im, bl, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
  std::printf("gauss7x7     hash=%016llx\n", (unsigned long long)matHash(bl));
  std::vector<cv::KeyPoint> fk; cv::FAST(im, fk, 20, true);
  uint64_t fh = 0; for (auto& k : fk) { int v[2] = {(int)k.pt.x, (int)k.pt.y}; fh = fnv1a((uint8_t*)v, sizeof v, fh ? fh : 1469598103934665603ULL); }
  std::printf("FAST(20)     n=%zu hash=%016llx\n", fk.size(), (unsigned long long)fh);

  ORB_SLAM3::ORBextractor ex(nFeatures, 1.2f, 8, 20, 7);
  std::vector<cv::KeyPoint> kps; cv::Mat desc; std::vector<int> lapping = {0, 1000};
  ex(im, cv::Mat(), kps, desc, lapping);
  for (int l = 0; l < (int)ex.mvImagePyramid.size(); ++l)
    std::printf("pyramid[%d]   %dx%d hash=%016llx\n", l, ex.mvImagePyramid[l].cols, ex.mvImagePyramid[l].rows,
                (unsigned long long)matHash(ex.mvImagePyramid[l]));
  double sx = 0, sy = 0, sa = 0;
  for (auto& k : kps) { sx += k.pt.x; sy += k.pt.y; sa += k.angle; }
  std::printf("ORB          n=%zu sum_x=%.4f sum_y=%.4f sum_angle=%.4f desc_hash=%016llx\n", kps.size(), sx, sy, sa,
              (unsigned long long)matHash(desc));
  for (int i = 0; i < 5 && i < (int)kps.size(); ++i)
    std::printf("  kp[%d] x=%.3f y=%.3f oct=%d angle=%.4f resp=%.6f\n", i, kps[i].pt.x, kps[i].pt.y, kps[i].octave,
                kps[i].angle, kps[i].response);
  // optional full dump: one line per keypoint "x y octave angle desc_hex"
  if (argc >= 5) {
    FILE* f = std::fopen(argv[4], "w");
    for (size_t i = 0; i < kps.size(); ++i) {
      std::fprintf(f, "%.4f %.4f %d %.6f ", kps[i].pt.x, kps[i].pt.y, kps[i].octave, kps[i].angle);
      for (int j = 0; j < desc.cols; ++j) std::fprintf(f, "%02x", desc.at<uint8_t>((int)i, j));
      std::fprintf(f, "\n");
    }
    std::fclose(f);
  }
  return 0;
}
