// Parity check of slam/orbslam3/include/WebScaleFilter.h against eval/vio_filter.py: replays a WebVIO run's live poses
// (trace.txt: frame, state, Twc; keyframes in the map) and an IMU file through the C++ filter as Tracking::WebScaleUpdate
// feeds it (tracked frames only, reset when the map changes), and writes per frame: t s sigma_ln_s tracked map
// (the vio_filter.py --out layout).
//   g++ -O2 -std=c++17 -I/usr/include/eigen3 -Islam/orbslam3/include eval/scale_filter_replay.cpp -o scale_filter_replay
//   scale_filter_replay <run dir> <sequence dir> <imu csv> <offset ms> <out>   (IMU time = camera time + offset)
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "WebScaleFilter.h"

using namespace ORB_SLAM3;

static std::vector<std::vector<double>> rows(const std::string& path, char sep) {
  std::vector<std::vector<double>> out;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<double> r;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, sep)) {
      if (c.empty()) continue;
      try { r.push_back(std::stod(c)); } catch (...) { r.push_back(NAN); }
    }
    out.push_back(r);
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 6) { std::cerr << "usage: scale_filter_replay <run> <seq> <imu csv> <offset ms> <out>\n"; return 1; }
  const std::string run = argv[1], seq = argv[2], imuPath = argv[3], out = argv[5];
  const double off = std::stod(argv[4]) * 1e-3;
  // camera -> body rotation from settings.yaml IMU.T_b_c1 (row-major 4x4 after "data:")
  std::ifstream sf(seq + "/settings.yaml");
  std::string txt((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
  const size_t p0 = txt.find('[', txt.find("data:", txt.find("IMU.T_b_c1")));
  const size_t p1 = txt.find(']', p0);
  std::vector<double> T;
  std::stringstream ts(txt.substr(p0 + 1, p1 - p0 - 1));
  std::string c;
  while (std::getline(ts, c, ',')) T.push_back(std::stod(c));
  Eigen::Matrix3d Rbc;
  for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) Rbc(r, k) = T[4 * r + k];
  // frame stamps (ns, kept exact) and IMU (ns, w, a)
  std::vector<long long> cam;
  {
    std::ifstream f(seq + "/mav0/cam0/data.csv");
    std::string line;
    while (std::getline(f, line)) if (!line.empty() && line[0] != '#') cam.push_back(std::stoll(line.substr(0, line.find(','))));
  }
  std::vector<double> ti;
  std::vector<Eigen::Vector3d> wi, ai;
  {
    std::ifstream f(imuPath);
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::stringstream ss(line);
      std::string v;
      std::vector<std::string> cols;
      while (std::getline(ss, v, ',')) cols.push_back(v);
      ti.push_back(std::stoll(cols[0]) * 1e-9);
      wi.emplace_back(std::stod(cols[1]), std::stod(cols[2]), std::stod(cols[3]));
      ai.emplace_back(std::stod(cols[4]), std::stod(cols[5]), std::stod(cols[6]));
    }
  }
  const auto tr = rows(run + "/trace.txt", ' ');
  WebScaleFilter F;
  if (const char* pr = std::getenv("PRUNE")) F.prune = std::atof(pr);  // test: the pruning threshold (header default otherwise)
  if (const char* rs = std::getenv("RESTART")) F.restart = std::atof(rs);  // test: bank lifetime s (a new bank every RESTART / 2)
  if (const char* sb = std::getenv("STABLEBACK")) F.stableBack = std::atof(sb);
  F.setRcb(Rbc.transpose());
  size_t k = 0;
  int mapid = 0;
  double prevKfs = 0;
  std::ofstream o(out);
  o << "# t s sigma_ln_s tracked map (C++ WebScaleFilter.h)\n";
  for (const auto& r : tr) {
    if (r.size() < 14) continue;
    const int fr = (int)r[0], state = (int)r[1];
    const double kfs = r[13];
    if (kfs == 0 && prevKfs > 0) { mapid++; F.reset(); F.setRcb(Rbc.transpose()); }
    prevKfs = kfs;
    const double t = cam[std::min<size_t>(fr, cam.size() - 1)] * 1e-9;  // camera clock
    const double tImuClock = t + off;                                    // the frame on the IMU clock
    while (k < ti.size() && ti[k] <= tImuClock + 1.0) { F.addImu(ti[k], ai[k], wi[k]); k++; }  // queued ahead, as live
    const bool ok = state == 2 || state == 5;
    if (ok) {
      const Eigen::Quaterniond q(r[9], r[6], r[7], r[8]);  // trace: qx qy qz qw
      F.addFrame(tImuClock, Eigen::Vector3d(r[3], r[4], r[5]), q.toRotationMatrix(), (int)kfs);
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "%.6f %.6f %.6f %d %d\n", t, F.valid || F.s > 0 ? F.s : 0.0, F.s > 0 ? F.relStd : -1.0, ok ? 1 : 0, mapid);
    o << buf;
  }
  return 0;
}
