// OpenVINS (MSCKF filter VIO; Geneva et al., ICRA 2020; https://github.com/rpng/open_vins, GPL-3.0) on a sequence in the
// EuRoC / phone-recording layout: mav0/cam0/data.csv (t ns, file) + mav0/cam0/data/<file>, mav0/imu0/data.csv (ASL: t ns,
// w rad/s, a m/s^2). Native reference for the synthetic set (claude/PLAN.md 4d, approach D): a tightly coupled filter on
// the same data as WebVIO. Offline, as fast as it runs (not real time). Feeding as OpenVINS's ROS player
// (ROS1Visualizer::callback_inertial): every IMU sample, then the queued frames older than that sample on the camera clock.
// Writes <out>/traj_tum.txt: the camera pose (TUM, Hamilton x y z w) at every frame after initialisation, keyed by the
// frame's own stamp; <out>/log.txt: frame index, initialised, time offset, update ms.
// Usage: ov_run <estimator_config.yaml> <sequence dir> <out dir>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "types/Vec.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

namespace fs = std::filesystem;

static std::vector<std::vector<std::string>> csv(const fs::path& p) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream f(p);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> r;
    std::stringstream ss(line);
    std::string c;
    while (std::getline(ss, c, ',')) r.push_back(c);
    rows.push_back(r);
  }
  return rows;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: ov_run <estimator_config.yaml> <sequence dir> <out dir>\n";
    return 1;
  }
  const fs::path seq = argv[2], out = argv[3];
  fs::create_directories(out);
  auto parser = std::make_shared<ov_core::YamlParser>(argv[1]);
  std::string verbosity = "WARNING";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);
  ov_msckf::VioManagerOptions params;
  params.print_and_load(parser);
  params.use_multi_threading_subs = false;
  if (!parser->successful()) {
    std::cerr << "unable to parse all parameters\n";
    return 1;
  }
  auto sys = std::make_shared<ov_msckf::VioManager>(params);

  struct Frame { double t; fs::path file; int idx; };
  std::vector<Frame> frames;
  int k = 0;
  for (auto& r : csv(seq / "mav0" / "cam0" / "data.csv"))
    frames.push_back({std::stoll(r[0]) * 1e-9, seq / "mav0" / "cam0" / "data" / r[1], k++});
  std::vector<ov_core::ImuData> imu;
  for (auto& r : csv(seq / "mav0" / "imu0" / "data.csv")) {
    ov_core::ImuData m;
    m.timestamp = std::stoll(r[0]) * 1e-9;
    m.wm << std::stod(r[1]), std::stod(r[2]), std::stod(r[3]);
    m.am << std::stod(r[4]), std::stod(r[5]), std::stod(r[6]);
    imu.push_back(m);
  }
  std::cout << "frames " << frames.size() << ", IMU samples " << imu.size() << std::endl;

  std::ofstream traj(out / "traj_tum.txt"), log(out / "log.txt");
  traj << "# t tx ty tz qx qy qz qw (camera, OpenVINS)\n" << std::setprecision(9) << std::fixed;
  log << "# frame initialised dt_cam_imu_ms update_ms\n";
  size_t next = 0;
  int initFrame = -1;
  double msTotal = 0;
  const auto wall0 = std::chrono::steady_clock::now();
  for (const auto& m : imu) {
    sys->feed_measurement_imu(m);
    const double tImuInC = m.timestamp - sys->get_state()->_calib_dt_CAMtoIMU->value()(0);
    while (next < frames.size() && frames[next].t < tImuInC) {
      const Frame& fr = frames[next++];
      cv::Mat im = cv::imread(fr.file.string(), cv::IMREAD_GRAYSCALE);
      if (im.empty()) { std::cerr << "cannot read " << fr.file << "\n"; continue; }
      ov_core::CameraData c;
      c.timestamp = fr.t;
      c.sensor_ids.push_back(0);
      c.images.push_back(im);
      c.masks.push_back(cv::Mat::zeros(im.rows, im.cols, CV_8UC1));
      const auto t0 = std::chrono::steady_clock::now();
      sys->feed_measurement_camera(c);
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      msTotal += ms;
      const bool init = sys->initialized();
      const auto st = sys->get_state();
      log << fr.idx << " " << (init ? 1 : 0) << " " << 1000 * st->_calib_dt_CAMtoIMU->value()(0) << " " << ms << "\n";
      if (!init) continue;
      if (initFrame < 0) initFrame = fr.idx;
      // JPL: quat_2_Rot(q_GtoI) = R_GtoI; calibration q_ItoC, p_IinC
      const Eigen::Matrix3d R_ItoG = ov_core::quat_2_Rot(st->_imu->quat()).transpose();
      const Eigen::Vector3d p_IinG = st->_imu->pos();
      const auto calib = st->_calib_IMUtoCAM.at(0);
      const Eigen::Matrix3d R_ItoC = ov_core::quat_2_Rot(calib->quat());
      const Eigen::Vector3d p_IinC = calib->pos();
      const Eigen::Matrix3d R_CtoG = R_ItoG * R_ItoC.transpose();
      const Eigen::Vector3d p_CinG = p_IinG + R_ItoG * (-R_ItoC.transpose() * p_IinC);
      const Eigen::Quaterniond q(R_CtoG);
      traj << fr.t << " " << p_CinG.x() << " " << p_CinG.y() << " " << p_CinG.z() << " " << q.x() << " " << q.y() << " "
           << q.z() << " " << q.w() << "\n";
    }
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  std::ofstream info(out / "run_info.txt");
  info << "frames " << frames.size() << "\nprocessed " << next << "\ninit_frame " << initFrame << "\nwall_s " << wall
       << "\nupdate_ms_mean " << (next ? msTotal / next : 0) << "\ndt_cam_imu_ms_final "
       << 1000 * sys->get_state()->_calib_dt_CAMtoIMU->value()(0) << "\n";
  std::cout << "done: " << next << " frames, initialised at frame " << initFrame << ", " << wall << " s" << std::endl;
  return 0;
}
