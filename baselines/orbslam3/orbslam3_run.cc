// ORB-SLAM3 baseline driver for the datasets used in this project (KITTI raw sync, TUM-VIE).
//
// Loads images / timestamps / IMU exactly like webar-vio's vio_run (same files, same clocks),
// runs ORB-SLAM3 (viewer off) and saves trajectories. ORB-SLAM3 itself is unmodified apart
// from the build-compatibility patches in scripts/setup_env.sh.
//
// Usage:
//   orbslam3_run --dataset kitti|tumvie --mode mono|mono_inertial --seq <dir>
//                --vocab ORBvoc.txt --settings <yaml> --out <dir> [--clahe]
//
// Outputs (in --out):
//   traj_tum.txt      all tracked frames of the largest map, TUM format, absolute dataset time (s)
//                     pose = Twc of camera (mono) or Twb of IMU body (mono_inertial; see ORB-SLAM3 SaveTrajectoryEuRoC)
//   kf_traj_tum.txt   keyframes only
//   track_times.txt   per-frame TrackMonocular wall time (ms)
//   run_info.txt

#include <System.h>
#include <ImuTypes.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using std::string;
using std::vector;

struct Imu { double t; double ax, ay, az, gx, gy, gz; };

static vector<string> lines(const fs::path& p) {
  std::ifstream f(p);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  vector<string> out; string l;
  while (std::getline(f, l)) {
    auto a = l.find_first_not_of(" \t\r");
    if (a == string::npos || l[a] == '#') continue;
    out.push_back(l.substr(a));
  }
  return out;
}

static vector<fs::path> listImages(const fs::path& dir) {
  vector<fs::path> out;
  for (auto& e : fs::directory_iterator(dir)) {
    auto ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());  // zero-padded names in both datasets
  return out;
}

static double kittiTime(const string& s) {
  int Y, M, D, hh, mm, ss; char frac[32] = {0};
  std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d.%31[0-9]", &Y, &M, &D, &hh, &mm, &ss, frac);
  std::tm tm{}; tm.tm_year = Y - 1900; tm.tm_mon = M - 1; tm.tm_mday = D;
  string f(frac); f = f.substr(0, 9); while (f.size() < 9) f.push_back('0');
  return (double)timegm(&tm) + hh * 3600.0 + mm * 60.0 + ss + std::stod("0." + f);
}

struct Data {
  vector<fs::path> images;
  vector<double> t;       // rebased seconds, t[i] belongs to images[i]
  vector<Imu> imu;        // rebased seconds
  double tBase = 0;
};

// Pair image files with timestamps by the frame index in the file name (files can be missing).
static void pairByIndex(const vector<fs::path>& files, const vector<double>& ts, Data& d) {
  for (const auto& f : files) {
    string digits;
    for (char c : f.stem().string()) if (std::isdigit((unsigned char)c)) digits.push_back(c);
    if (digits.empty()) continue;
    const long k = std::stol(digits);
    if (k >= 0 && k < (long)ts.size()) { d.images.push_back(f); d.t.push_back(ts[k]); }
  }
  if (d.images.size() < ts.size())
    std::cerr << "WARNING: " << ts.size() - d.images.size() << " frames have no image file; skipped\n";
}

static Data loadKitti(const fs::path& seq) {
  Data d;
  vector<double> ti, to;
  for (auto& l : lines(seq / "image_02" / "timestamps.txt")) ti.push_back(kittiTime(l));
  for (auto& l : lines(seq / "oxts" / "timestamps.txt")) to.push_back(kittiTime(l));
  d.tBase = std::min(ti.front(), to.front());
  for (auto& x : ti) x -= d.tBase;
  pairByIndex(listImages(seq / "image_02" / "data"), ti, d);
  // (KITTI sync OXTS is ~10 Hz: not usable for mono-inertial; IMU intentionally not loaded.)
  return d;
}

static Data loadTumvie(const fs::path& seq) {
  Data d;
  cv::FileStorage fsCal((seq / "camera-calibration.json").string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  cv::FileNode root = fsCal["value0"].empty() ? fsCal.root() : fsCal["value0"];
  const double camOff = root["cam_time_offset_ns"].empty() ? 0.0 : (double)root["cam_time_offset_ns"] * 1e-9;
  vector<double> ti;
  for (auto& l : lines(seq / "left_images" / "image_timestamps_left.txt")) ti.push_back(std::stod(l) * 1e-6 + camOff);
  d.tBase = ti.front();
  for (auto& x : ti) x -= d.tBase;
  pairByIndex(listImages(seq / "left_images"), ti, d);
  for (auto& l : lines(seq / "imu_data.txt")) {
    std::stringstream ss(l); double v[7];
    for (double& x : v) ss >> x;
    if (!ss) continue;
    // imu_data.txt: t_us gx gy gz ax ay az
    d.imu.push_back({v[0] * 1e-6 - d.tBase, v[4], v[5], v[6], v[1], v[2], v[3]});
  }
  return d;
}

// ORB-SLAM3 writes EuRoC-style files with timestamp in ns (of the rebased time); convert to TUM seconds (absolute).
static void euRoCToTum(const fs::path& in, const fs::path& out, double tBase) {
  std::ifstream f(in);
  std::ofstream o(out);
  o << std::setprecision(17) << "# t tx ty tz qx qy qz qw\n";
  string l;
  while (std::getline(f, l)) {
    std::stringstream ss(l); double t, x, y, z, qx, qy, qz, qw;
    if (!(ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw)) continue;
    o << (t * 1e-9 + tBase) << " " << x << " " << y << " " << z << " " << qx << " " << qy << " " << qz << " " << qw << "\n";
  }
}

int main(int argc, char** argv) {
  string dataset, mode, seq, vocab, settings, out;
  bool clahe = false;
  for (int i = 1; i < argc; ++i) {
    string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); } return string(argv[++i]); };
    if (a == "--dataset") dataset = next();
    else if (a == "--mode") mode = next();
    else if (a == "--seq") seq = next();
    else if (a == "--vocab") vocab = next();
    else if (a == "--settings") settings = next();
    else if (a == "--out") out = next();
    else if (a == "--clahe") clahe = true;
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }
  if (dataset.empty() || mode.empty() || seq.empty() || vocab.empty() || settings.empty() || out.empty()) {
    std::cerr << "usage: orbslam3_run --dataset kitti|tumvie --mode mono|mono_inertial --seq DIR --vocab V --settings S --out DIR [--clahe]\n";
    return 2;
  }
  const bool inertial = (mode == "mono_inertial");
  if (inertial && dataset == "kitti") { std::cerr << "KITTI sync OXTS (~10 Hz) is not usable for mono-inertial\n"; return 2; }

  Data d = (dataset == "kitti") ? loadKitti(seq) : loadTumvie(seq);
  const int N = (int)std::min(d.images.size(), d.t.size());
  fs::create_directories(out);

  ORB_SLAM3::System SLAM(vocab, settings,
                         inertial ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::MONOCULAR,
                         /*bUseViewer=*/false);
  const float imageScale = SLAM.GetImageScale();
  cv::Ptr<cv::CLAHE> claheOp = cv::createCLAHE(3.0, cv::Size(8, 8));  // as in ORB-SLAM3's TUM-VI examples

  // First IMU sample considered: the last one at or before the first frame (as in mono_inertial_tum_vi.cc)
  size_t imuIdx = 0;
  if (inertial) {
    while (imuIdx < d.imu.size() && d.imu[imuIdx].t <= d.t[0]) ++imuIdx;
    if (imuIdx > 0) --imuIdx;
  }

  std::ofstream times(fs::path(out) / "track_times.txt");
  vector<ORB_SLAM3::IMU::Point> meas;
  for (int i = 0; i < N; ++i) {
    cv::Mat im = cv::imread(d.images[i].string(), cv::IMREAD_GRAYSCALE);
    if (im.empty()) { std::cerr << "failed to read " << d.images[i] << "\n"; return 1; }
    if (clahe) claheOp->apply(im, im);
    if (imageScale != 1.f) cv::resize(im, im, cv::Size(im.cols * imageScale, im.rows * imageScale));

    meas.clear();
    if (inertial && i > 0) {
      while (imuIdx < d.imu.size() && d.imu[imuIdx].t <= d.t[i]) {
        const auto& m = d.imu[imuIdx++];
        meas.emplace_back(m.ax, m.ay, m.az, m.gx, m.gy, m.gz, m.t);
      }
    }
    auto t0 = std::chrono::steady_clock::now();
    SLAM.TrackMonocular(im, d.t[i], meas);
    auto t1 = std::chrono::steady_clock::now();
    times << i << " " << std::chrono::duration<double, std::milli>(t1 - t0).count()
          << " " << SLAM.GetTrackingState() << "\n";
    if (i % 200 == 0) std::cerr << "frame " << i << "/" << N - 1 << " state=" << SLAM.GetTrackingState() << "\n";
  }

  SLAM.Shutdown();
  const fs::path rawF = fs::path(out) / "f_euroc_ns.txt", rawKF = fs::path(out) / "kf_euroc_ns.txt";
  SLAM.SaveTrajectoryEuRoC(rawF.string());
  SLAM.SaveKeyFrameTrajectoryEuRoC(rawKF.string());
  euRoCToTum(rawF, fs::path(out) / "traj_tum.txt", d.tBase);
  euRoCToTum(rawKF, fs::path(out) / "kf_traj_tum.txt", d.tBase);

  std::ofstream info(fs::path(out) / "run_info.txt");
  info << std::setprecision(17) << "dataset " << dataset << "\nmode " << mode << "\nseq " << seq
       << "\nsettings " << settings << "\nclahe " << clahe << "\nframes " << N << "\nt_base " << d.tBase << "\n";
  return 0;
}
