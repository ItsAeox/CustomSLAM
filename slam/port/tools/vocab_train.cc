// Train a DBoW2 vocabulary (ORB-SLAM3's DBoW2 fork, k-means++ hierarchical) from images, with descriptors computed by
// ORB-SLAM3's own ORBextractor (oriented FAST + octree, as in tracking), either ORB (rotated BRIEF) or TEBLID-256.
// Training ORB and TEBLID vocabularies with the same images and parameters isolates the descriptor in comparisons.
//
// Usage: vocab_train <out.bin> <orb|teblid> <k> <L> <every_nth_image> [--clahe] [--features N] [--hbrb] <image_dir> [...]
//   --hbrb: real-valued k-means centres while clustering, binarised at the end (HBRBVocabulary below)
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

// --hbrb: HBRB-style training (after Lee et al. 2026, "HBRB-BoW", arXiv 2603.04144; re-implemented, their code is not
// public). DBoW2's k-majority k-means assigns descriptors by Hamming distance to binarised centres at every
// iteration, which loses the centres' precision. Here each node's k-means keeps real-valued centres (per-bit means)
// while clustering, with squared Euclidean distance between a descriptor's bits and a centre; only the final centres are
// binarised (>= 0.5, the bitwise majority of the members). Same k-means++ seeding, same tree shape (k, L). Word
// weights come from DBoW2's setNodeWeights, which sends the training descriptors down the binarised tree with Hamming
// distance: the same descent as at runtime, so training and use agree.
class HBRBVocabulary : public ORB_SLAM3::ORBVocabulary {
 public:
  HBRBVocabulary(int k, int L) : ORB_SLAM3::ORBVocabulary(k, L, DBoW2::TF_IDF, DBoW2::L1_NORM) {}
  void createHBRB(const std::vector<std::vector<cv::Mat>>& features) {
    m_nodes.clear();
    m_words.clear();
    m_nodes.reserve(nodesFor(m_k, m_L));
    m_nodes.push_back(Node(0));  // root
    std::vector<pDescriptor> all;
    getFeatures(features, all);
    step(0, all, 1);
    createWords();
    setNodeWeights(features);
  }

 private:
  static size_t nodesFor(int k, int L) { size_t n = 1, p = 1; for (int l = 0; l < L; ++l) { p *= k; n += p; } return n; }
  static int bit(const unsigned char* d, int i) { return (d[i >> 3] >> (i & 7)) & 1; }

  void step(NodeId parent, const std::vector<pDescriptor>& descs, int level) {
    if (descs.empty()) return;
    const int n = (int)descs.size();
    std::vector<cv::Mat> centres;            // final binary centres
    std::vector<std::vector<int>> groups;    // members per cluster
    if (n <= m_k) {
      for (int i = 0; i < n; ++i) { centres.push_back(descs[i]->clone()); groups.push_back({i}); }
    } else {
      std::vector<cv::Mat> seeds;
      initiateClustersKMpp(descs, seeds);  // DBoW2's k-means++ (Hamming), as standard training
      const int k = (int)seeds.size();
      std::vector<std::vector<float>> c(k, std::vector<float>(256));
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < 256; ++i) c[j][i] = (float)bit(seeds[j].ptr<unsigned char>(), i);
      std::vector<int> assign(n, -1);
      std::vector<float> table((size_t)k * 32 * 256);
      for (int it = 0; it < 30; ++it) {
        // per centre: sum of the centre values over the set bits of each possible byte value, per byte position
        std::vector<float> norm2(k, 0.f);
        for (int j = 0; j < k; ++j) {
          for (int i = 0; i < 256; ++i) norm2[j] += c[j][i] * c[j][i];
          for (int b = 0; b < 32; ++b)
            for (int v = 0; v < 256; ++v) {
              float s = 0.f;
              for (int q = 0; q < 8; ++q) if ((v >> q) & 1) s += c[j][8 * b + q];
              table[((size_t)j * 32 + b) * 256 + v] = s;
            }
        }
        bool changed = false;
        for (int d = 0; d < n; ++d) {
          const unsigned char* p = descs[d]->ptr<unsigned char>();
          int pop = 0;
          for (int b = 0; b < 32; ++b) pop += __builtin_popcount(p[b]);
          int best = 0;
          float bestDist = 1e30f;
          for (int j = 0; j < k; ++j) {  // |x - c|^2 = popcount(x) - 2 sum_{x_i = 1} c_i + |c|^2
            const float* t = &table[(size_t)j * 32 * 256];
            float dot = 0.f;
            for (int b = 0; b < 32; ++b) dot += t[b * 256 + p[b]];
            const float dist = pop - 2.f * dot + norm2[j];
            if (dist < bestDist) { bestDist = dist; best = j; }
          }
          if (assign[d] != best) { assign[d] = best; changed = true; }
        }
        // new real-valued centres: per-bit means of the members (an empty cluster keeps its centre)
        std::vector<std::vector<double>> sum(k, std::vector<double>(256, 0.0));
        std::vector<int> cnt(k, 0);
        for (int d = 0; d < n; ++d) {
          const unsigned char* p = descs[d]->ptr<unsigned char>();
          cnt[assign[d]]++;
          for (int i = 0; i < 256; ++i) sum[assign[d]][i] += bit(p, i);
        }
        for (int j = 0; j < k; ++j)
          if (cnt[j]) for (int i = 0; i < 256; ++i) c[j][i] = (float)(sum[j][i] / cnt[j]);
        if (!changed) break;
      }
      groups.assign(k, {});
      for (int d = 0; d < n; ++d) groups[assign[d]].push_back(d);
      for (int j = 0; j < k; ++j) {  // binarise: bitwise majority of the members
        cv::Mat m = cv::Mat::zeros(1, 32, CV_8U);
        unsigned char* q = m.ptr<unsigned char>();
        for (int i = 0; i < 256; ++i) if (c[j][i] >= 0.5f) q[i >> 3] |= (unsigned char)(1 << (i & 7));
        centres.push_back(m);
      }
    }
    // all children first, then the next level (DBoW2's node order); ids are indices (m_nodes may reallocate)
    std::vector<std::pair<NodeId, int>> kids;
    for (size_t j = 0; j < centres.size(); ++j) {
      if (groups[j].empty()) continue;
      NodeId id = m_nodes.size();
      m_nodes.push_back(Node(id));
      m_nodes.back().descriptor = centres[j];
      m_nodes.back().parent = parent;
      m_nodes[parent].children.push_back(id);
      kids.push_back({id, (int)j});
    }
    if (level >= m_L) return;
    for (const auto& [id, j] : kids) {
      if (groups[j].size() <= 1) continue;
      std::vector<pDescriptor> sub;
      sub.reserve(groups[j].size());
      for (int d : groups[j]) sub.push_back(descs[d]);
      step(id, sub, level + 1);
    }
  }
};

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr, "usage: vocab_train <out.bin> <orb|teblid> <k> <L> <every_nth> [--clahe] [--features N] <dir> [...]\n");
    return 2;
  }
  const std::string out = argv[1], kind = argv[2];
  const int k = std::stoi(argv[3]), L = std::stoi(argv[4]), step = std::max(1, std::stoi(argv[5]));
  bool clahe = false, hbrb = false;
  int nFeatures = 1000;
  std::vector<std::string> dirs;
  for (int i = 6; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--clahe") clahe = true;
    else if (a == "--hbrb") hbrb = true;
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

  HBRBVocabulary voc(k, L);
  if (hbrb) voc.createHBRB(features);  // real-valued k-means centres, binarised at the end
  else voc.create(features);           // DBoW2's standard k-majority training
  const auto t2 = std::chrono::steady_clock::now();
  std::printf("vocabulary: %u words (%.0f s)\n", voc.size(), std::chrono::duration<double>(t2 - t1).count());
  if (!voc.saveToBinaryFile(out)) { std::fprintf(stderr, "save failed: %s\n", out.c_str()); return 1; }
  std::printf("saved %s\n", out.c_str());
  return 0;
}
