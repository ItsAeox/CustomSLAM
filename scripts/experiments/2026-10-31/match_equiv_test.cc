// 2 Nov: WebPriorFreePnP before 14733bd vs after (gathered descriptors, 64-bit popcounts, keypoint ranges on 1 / 4 threads) on
// random descriptors: identical best / second best / match for every keypoint. g++ -O2 -pthread match_equiv_test.cc
// old (8 x 32-bit, point-outer over all keypoints) vs new (gathered, 64-bit, keypoint ranges) matching: identical?
#include <cstdint>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>
#include <algorithm>
using namespace std;
int main() {
  mt19937 rng(7);
  for (int trial = 0; trial < 20; trial++) {
    const int N = 800 + rng() % 600, P = 2000 + rng() % 4000;
    vector<uint64_t> F(N * 4);
    for (auto& x : F) x = ((uint64_t)rng() << 32) | rng();
    vector<vector<vector<uint64_t>>> pts(P);
    for (int j = 0; j < P; j++) {
      int nd = rng() % 7;  // 0..6 descriptors (0 = skipped)
      for (int d = 0; d < nd; d++) {
        vector<uint64_t> v(4);
        if (rng() % 3 == 0) { int i = rng() % N; for (int k = 0; k < 4; k++) v[k] = F[i * 4 + k] ^ (1ull << (rng() % 64)); }  // near a keypoint
        else for (auto& x : v) x = ((uint64_t)rng() << 32) | rng();
        pts[j].push_back(v);
      }
    }
    // old
    vector<int> a1(N, 256), a2(N, 256), am(N, -1);
    for (int j = 0; j < P; j++) {
      if (pts[j].empty()) continue;
      for (int i = 0; i < N; i++) {
        const uint32_t* b = (const uint32_t*)&F[i * 4];
        int dist = 256;
        for (auto& v : pts[j]) { const uint32_t* a = (const uint32_t*)v.data(); int d = 0; for (int k = 0; k < 8; k++) d += __builtin_popcount(a[k] ^ b[k]); dist = min(dist, d); }
        if (dist < a1[i]) { a2[i] = a1[i]; a1[i] = dist; am[i] = j; } else if (dist < a2[i]) a2[i] = dist;
      }
    }
    // new
    vector<int> vP, vStart(1, 0); vector<const uint64_t*> vDesc;
    for (int j = 0; j < P; j++) { size_t n0 = vDesc.size(); for (auto& v : pts[j]) vDesc.push_back(v.data()); if (vDesc.size() == n0) continue; vP.push_back(j); vStart.push_back(vDesc.size()); }
    for (int nt : {1, 4}) {
      vector<int> b1(N, 256), b2(N, 256), bm(N, -1);
      auto match = [&](int a, int b) {
        for (size_t j = 0; j + 1 < vStart.size(); j++)
          for (int i = a; i < b; i++) {
            const uint64_t* f = &F[i * 4]; int dist = 256;
            for (int k = vStart[j]; k < vStart[j + 1]; k++) { const uint64_t* d = vDesc[k];
              dist = min(dist, __builtin_popcountll(d[0] ^ f[0]) + __builtin_popcountll(d[1] ^ f[1]) + __builtin_popcountll(d[2] ^ f[2]) + __builtin_popcountll(d[3] ^ f[3])); }
            if (dist < b1[i]) { b2[i] = b1[i]; b1[i] = dist; bm[i] = vP[j]; } else if (dist < b2[i]) b2[i] = dist;
          }
      };
      vector<thread> th; for (int t = 0; t < nt; t++) th.emplace_back(match, N * t / nt, N * (t + 1) / nt); for (auto& x : th) x.join();
      int diff = 0; for (int i = 0; i < N; i++) diff += (a1[i] != b1[i]) || (a2[i] != b2[i]) || (am[i] != bm[i]);
      printf("trial %d N %d P %d threads %d: differing keypoints %d\n", trial, N, P, nt, diff);
    }
  }
}
