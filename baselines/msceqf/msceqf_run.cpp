// MSCEqF (Multi State Constraint Equivariant Filter; Fornasier et al., RA-L 2023; https://github.com/aau-cns/MSCEqF,
// Apache-2.0) on a sequence in the EuRoC / phone-recording layout: mav0/cam0/data.csv (t ns, file) + mav0/cam0/data/<file>,
// mav0/imu0/data.csv (ASL: t ns, w rad/s, a m/s^2). Native reference for the synthetic set and the phone recordings
// (claude/PLAN.md 4d): an equivariant MSCKF-type filter on the same data as WebVIO. Offline, as fast as it runs. Fed as its
// EuRoC example (every sample in time order); MSCEqF has no time-offset state, so frame stamps are moved to the IMU clock
// by the camera lag (stamp - lag; phone-like IMU 0.088 s, as EqVIO's main:cameraLag).
// Writes <out>/traj_camclock.txt: the camera pose T_wc = T_wi S (S: the online camera extrinsic) at every frame after
// initialisation, keyed by the frame's own (camera-clock) stamp; <out>/timing.csv: frame, initialised, camera update ms (image already
// decoded: tracking + filter update), IMU propagation ms since the previous frame.
// Usage: msceqf_run <config.yaml> <sequence dir> <camera lag s> <out dir>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>

#include "msceqf/msceqf.hpp"

namespace {
struct Row {
  long long ns;
  std::vector<std::string> cols;
};

std::vector<Row> readCsv(const std::string& path) {
  std::vector<Row> rows;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    Row r;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) r.cols.push_back(c);
    r.ns = std::stoll(r.cols[0]);
    rows.push_back(r);
  }
  return rows;
}

double ms(std::chrono::steady_clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: msceqf_run <config.yaml> <sequence dir> <camera lag s> <out dir>\n";
    return 1;
  }
  const std::string cfg = argv[1], seq = argv[2], out = argv[4];
  const double lag = std::stod(argv[3]);
  const auto imu = readCsv(seq + "/mav0/imu0/data.csv");
  const auto cam = readCsv(seq + "/mav0/cam0/data.csv");
  msceqf::MSCEqF sys(cfg);
  std::ofstream traj(out + "/traj_camclock.txt"), timing(out + "/timing.csv");
  traj << "# t tx ty tz qx qy qz qw (camera, MSCEqF; camera clock)\n";
  timing << "frame,initialised,camera_ms,imu_ms\n";
  traj << std::setprecision(9) << std::fixed;
  size_t i = 0;
  double imuMs = 0;
  for (size_t k = 0; k < cam.size(); k++) {
    const double tc = cam[k].ns * 1e-9 - lag;
    for (; i < imu.size() && imu[i].ns * 1e-9 <= tc; i++) {
      msceqf::Imu m;
      m.timestamp_ = imu[i].ns * 1e-9;
      m.ang_ << std::stod(imu[i].cols[1]), std::stod(imu[i].cols[2]), std::stod(imu[i].cols[3]);
      m.acc_ << std::stod(imu[i].cols[4]), std::stod(imu[i].cols[5]), std::stod(imu[i].cols[6]);
      const auto t0 = std::chrono::steady_clock::now();
      sys.processMeasurement(m);
      imuMs += ms(std::chrono::steady_clock::now() - t0);
    }
    msceqf::Camera c;
    c.timestamp_ = tc;
    c.image_ = cv::imread(seq + "/mav0/cam0/data/" + cam[k].cols[1], cv::IMREAD_GRAYSCALE);
    if (c.image_.empty()) continue;
    const auto t0 = std::chrono::steady_clock::now();
    sys.processMeasurement(c);
    const double camMs = ms(std::chrono::steady_clock::now() - t0);
    const bool init = sys.isInit();
    timing << k << "," << init << "," << camMs << "," << imuMs << "\n";
    imuMs = 0;
    if (!init) continue;
    const auto& x = sys.stateEstimate();
    const Eigen::Matrix4d Twc = x.P().asMatrix().cast<double>() * x.S().asMatrix().cast<double>();
    const Eigen::Quaterniond q(Eigen::Matrix3d(Twc.block<3, 3>(0, 0)));
    traj << cam[k].ns * 1e-9 << " " << Twc(0, 3) << " " << Twc(1, 3) << " " << Twc(2, 3) << " " << q.x() << " " << q.y() << " " << q.z()
         << " " << q.w() << "\n";
  }
  std::cout << out << ": " << cam.size() << " frames, " << imu.size() << " IMU samples\n";
  return 0;
}
