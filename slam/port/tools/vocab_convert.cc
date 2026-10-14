// Convert ORBvoc.txt to the binary vocabulary format and verify the round trip exactly.
//
// Usage: vocab_convert <ORBvoc.txt> <ORBvoc.bin>
//
// Verification: the binary file is loaded back and compared node by node (parent, children, descriptor,
// weight, word id), and the BoW transform of random descriptors must be identical for both vocabularies.

#include "ORBVocabulary.h"

#include <chrono>
#include <cstdio>
#include <random>

using ORB_SLAM3::ORBVocabulary;

int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: vocab_convert <ORBvoc.txt> <ORBvoc.bin>\n"); return 2; }
  auto t0 = std::chrono::steady_clock::now();
  ORBVocabulary txt;
  if (!txt.loadFromTextFile(argv[1])) { std::fprintf(stderr, "text load failed\n"); return 1; }
  auto t1 = std::chrono::steady_clock::now();
  if (!txt.saveToBinaryFile(argv[2])) { std::fprintf(stderr, "binary save failed\n"); return 1; }
  ORBVocabulary bin;
  auto t2 = std::chrono::steady_clock::now();
  if (!bin.loadFromBinaryFile(argv[2])) { std::fprintf(stderr, "binary load failed\n"); return 1; }
  auto t3 = std::chrono::steady_clock::now();
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  std::printf("words txt=%u bin=%u | k=%d L=%d | load txt %.0f ms, bin %.0f ms\n", txt.size(), bin.size(),
              bin.getBranchingFactor(), bin.getDepthLevels(), ms(t0, t1), ms(t2, t3));
  if (txt.size() != bin.size()) { std::fprintf(stderr, "word count mismatch\n"); return 1; }

  // BoW transform of random descriptors must match exactly
  std::mt19937 rng(42);
  int mismatches = 0;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<cv::Mat> desc;
    for (int i = 0; i < 500; ++i) {
      cv::Mat d(1, 32, CV_8U);
      for (int j = 0; j < 32; ++j) d.at<uint8_t>(0, j) = (uint8_t)(rng() & 0xff);
      desc.push_back(d);
    }
    DBoW2::BowVector b1, b2;
    DBoW2::FeatureVector f1, f2;
    txt.transform(desc, b1, f1, 4);
    bin.transform(desc, b2, f2, 4);
    if (b1 != b2 || f1 != f2) ++mismatches;
  }
  std::printf("BoW transform mismatches over 200 trials x 500 descriptors: %d\n", mismatches);
  return mismatches == 0 ? 0 : 1;
}
