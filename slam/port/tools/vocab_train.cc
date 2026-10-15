// Train a DBoW2 vocabulary (ORB-SLAM3's DBoW2 fork, k-means++ hierarchical) from images, with descriptors computed by
// ORB-SLAM3's own ORBextractor (oriented FAST + octree, as in tracking), either ORB (rotated BRIEF) or TEBLID-256.
// Training ORB and TEBLID vocabularies with the same images and parameters isolates the descriptor in comparisons.
//
// Usage: vocab_train <out.bin> <orb|teblid> <k> <L> <every_nth_image> [--clahe] [--features N] <image_dir> [...]
//   image_dir: a folder of .png/.jpg images (sorted; every n-th one is used).
// Output: the binary vocabulary format (DBOW2BIN v1, see ORBVocabulary / TemplatedVocabulary::saveToBinaryFile).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "WebConfig.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: vocab_train <out.bin> <orb|teblid> <k> <L> <every_nth> [--clahe] [--features N] <dir> [...]\n");
    return 2;
  }
  const std::string out = argv[1], kind = argv[2];
  const int k = std::stoi(argv[3]), L = std::stoi(argv[4]), step = std::max(1, std::stoi(argv[5]));
  bool clahe = false;
  int nFeatures = 1000;
  std::vector<std::string> dirs;
  for (int i = 6; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--clahe") clahe = true;
    else if (a == "--features" && i + 1 < argc) nFeatures = std::stoi(argv[++i]);
    else dirs.push_back(a);
  }
  ORB_SLAM3::webConfig().descriptor = (kind == "teblid") ? 1 : 0;
  ORB_SLAM3::ORBextractor extractor(nFeatures, 1.2f, 8, 20, 7);  // ORB-SLAM3's extractor, tracking defaults
  cv::Ptr<cv::CLAHE> claheOp = cv::createCLAHE(3.0, cv::Size(8, 8));

  std::vector<std::vector<cv::Mat>> features;
  size_t nDesc = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (const std::string& d : dirs) {
    std::vector<fs::path> imgs;
    for (const auto& e : fs::directory_iterator(d)) {
      const std::string ext = e.path().extension().string();
      if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") imgs.push_back(e.path());
    }
    std::sort(imgs.begin(), imgs.end());
    size_t used = 0;
    for (size_t i = 0; i < imgs.size(); i += step) {
      cv::Mat im = cv::imread(imgs[i].string(), cv::IMREAD_GRAYSCALE);
      if (im.empty()) continue;
      if (clahe) claheOp->apply(im, im);
      std::vector<cv::KeyPoint> kps;
      cv::Mat desc;
      std::vector<int> lapping = {0, 0};
      extractor(im, cv::Mat(), kps, desc, lapping);
      std::vector<cv::Mat> f;
      f.reserve(desc.rows);
      for (int r = 0; r < desc.rows; ++r) f.push_back(desc.row(r).clone());
      nDesc += f.size();
      features.push_back(std::move(f));
      ++used;
    }
    std::printf("%s: %zu of %zu images\n", d.c_str(), used, imgs.size());
    std::fflush(stdout);
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("%zu images, %zu %s descriptors (%.0f s). Training k=%d L=%d ...\n", features.size(), nDesc,
              kind.c_str(), std::chrono::duration<double>(t1 - t0).count(), k, L);
  std::fflush(stdout);

  ORB_SLAM3::ORBVocabulary voc(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM);
  voc.create(features);
  const auto t2 = std::chrono::steady_clock::now();
  std::printf("vocabulary: %u words (%.0f s)\n", voc.size(), std::chrono::duration<double>(t2 - t1).count());
  if (!voc.saveToBinaryFile(out)) { std::fprintf(stderr, "save failed: %s\n", out.c_str()); return 1; }
  std::printf("saved %s\n", out.c_str());
  return 0;
}
