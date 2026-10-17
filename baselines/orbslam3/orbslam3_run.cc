// ORB-SLAM3 baseline driver for the datasets used in this project (KITTI raw sync, TUM-VIE).
//
// Loads images / timestamps / IMU exactly like webar-vio's vio_run (same files, same clocks),
// runs ORB-SLAM3 (viewer off) and saves trajectories. ORB-SLAM3 itself is unmodified apart
// from the build-compatibility patches in scripts/setup_env.sh.
//
// Usage:
//   orbslam3_run --dataset kitti|tumvie|euroc|tumvi --mode mono|mono_inertial --seq <dir>
//                --vocab ORBvoc.txt --settings <yaml> --out <dir> [--clahe]
//
// Outputs (in --out):
//   traj_tum.txt      all tracked frames of the largest map, TUM format, absolute dataset time (s)
//                     pose = Twc of camera (mono) or Twb of IMU body (mono_inertial; see ORB-SLAM3 SaveTrajectoryEuRoC)
//   kf_traj_tum.txt   keyframes only
//   traj_map<id>_tum.txt  all tracked frames of each map with keyframes (multi-map runs; eval/evaluate_multimap.py)
//   track_times.txt   per-frame TrackMonocular wall time (ms)
//   run_info.txt      includes frames processed / dropped and the effective playback rate (sequence time / wall time)
//
// Pacing (see ISSUES H1): --realtime is the protocol of ORB-SLAM3's examples (after each frame sleep for the frame
// period minus the TrackMonocular time). It ignores image loading, which on WSL's /mnt drives made runs play at
// ~0.5x. --rt-wall plays every frame on a wall-clock schedule while a background thread loads the next one;
// --live behaves like a camera (the tracker always gets the newest frame, late frames are dropped).

#include <System.h>
#include <ImuTypes.h>
#include "Thirdparty/DBoW2/DUtils/Random.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Atlas.h"
#include "Map.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/heap.h>
#else
#include <sys/resource.h>
#endif

// Memory of this process (MB): WASM heap size (grows, never shrinks) or the native peak resident set.
static double memoryMB() {
#ifdef __EMSCRIPTEN__
  return emscripten_get_heap_size() / 1048576.0;
#else
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss / 1024.0;  // KB on Linux
#endif
}

// Map size over all maps of the atlas: [map points, keyframes, maps]
static std::array<long, 3> mapSize(ORB_SLAM3::System& slam) {
  std::array<long, 3> r{0, 0, 0};
#ifdef ORBSLAM3_WEB
  for (ORB_SLAM3::Map* m : slam.WebAtlas()->GetAllMaps()) {
    r[0] += (long)m->MapPointsInMap();
    r[1] += (long)m->KeyFramesInMap();
    ++r[2];
  }
#endif
  return r;
}

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

// EuRoC / TUM-VI "EuRoC format" (ASL layout): <seq>/mav0/cam0/data.csv "#timestamp [ns],filename",
// images in mav0/cam0/data/, IMU in mav0/imu0/data.csv "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z".
// Same frame list as ORB-SLAM3's mono_euroc / mono_tum_vi examples (their timestamp lists = every cam0 frame).
static Data loadAsl(const fs::path& seq) {
  Data d;
  const fs::path cam = seq / "mav0" / "cam0";
  vector<std::pair<long long, string>> rows;
  for (auto& l : lines(cam / "data.csv")) {
    auto c = l.find(',');
    if (c == string::npos) continue;
    string name = l.substr(c + 1);
    while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();
    rows.emplace_back(std::stoll(l.substr(0, c)), name);
  }
  if (rows.empty()) throw std::runtime_error("no frames in " + (cam / "data.csv").string());
  const long long ns0 = rows.front().first;  // rebase in integer ns: exact
  d.tBase = (double)ns0 * 1e-9;
  for (auto& [ns, name] : rows) {
    const fs::path p = cam / "data" / name;
    if (!fs::exists(p)) continue;
    d.images.push_back(p);
    d.t.push_back((double)(ns - ns0) * 1e-9);
  }
  if (d.images.size() < rows.size())
    std::cerr << "WARNING: " << rows.size() - d.images.size() << " frames have no image file; skipped\n";
  for (auto& l : lines(seq / "mav0" / "imu0" / "data.csv")) {
    std::stringstream ss(l);
    string tok; vector<string> v;
    while (std::getline(ss, tok, ',')) v.push_back(tok);
    if (v.size() < 7) continue;
    const double t = (double)(std::stoll(v[0]) - ns0) * 1e-9;
    d.imu.push_back({t, std::stod(v[4]), std::stod(v[5]), std::stod(v[6]), std::stod(v[1]), std::stod(v[2]), std::stod(v[3])});
  }
  return d;
}

// Whole-file read + imdecode: cv::imread's small buffered reads cost ~23 ms per EuRoC PNG on WSL's 9P mount (/mnt/h)
// against ~7 ms for one large read.
static cv::Mat readGray(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const std::streamsize n = f.tellg();
  if (n <= 0) return {};
  std::vector<uchar> buf((size_t)n);
  f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(buf.data()), n)) return {};
  return cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
}

// Frames for the tracker, loaded on a background thread.
//  - in order (rt-wall): every frame, the next one loaded while the current one is tracked;
//  - camera (live): frame k "arrives" at wall time rate * (t[k] - t[0]); the thread keeps loading the newest arrived
//    frame, and the tracker always takes the newest loaded one, so frames that arrive while it is busy are dropped.
class FrameSource {
 public:
  FrameSource(std::function<cv::Mat(int)> load, const vector<double>& t, int n, double rate, bool camera)
      : load_(std::move(load)), t_(t), n_(n), rate_(rate), camera_(camera), wall0_(std::chrono::steady_clock::now()) {
    th_ = std::thread([this] { run(); });
  }
  ~FrameSource() {
    { std::lock_guard<std::mutex> l(m_); stop_ = true; }
    cv_.notify_all();
    th_.join();
  }
  // Blocks until a frame newer than the last one taken is ready; returns its index (or -1 at the end).
  int next(cv::Mat& im) {
    std::unique_lock<std::mutex> l(m_);
    cv_.wait(l, [this] { return readyIdx_ > taken_ || (done_ && readyIdx_ <= taken_); });
    if (readyIdx_ <= taken_) return -1;
    taken_ = readyIdx_;
    im = ready_;
    ready_ = cv::Mat();
    cv_.notify_all();
    return taken_;
  }
  double arrival(int k) const { return rate_ * (t_[k] - t_[0]); }  // seconds after start
  double now() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0_).count(); }

 private:
  void run() {
    int k = 0;
    while (true) {
      if (camera_) {  // newest frame that has arrived (wait for the next one if none is new)
        double tn = now();
        while (k + 1 < n_ && arrival(k + 1) <= tn) ++k;
        { std::lock_guard<std::mutex> l(m_); if (stop_) return; }
        if (k <= loaded_) {
          if (k + 1 >= n_) break;
          std::this_thread::sleep_for(std::chrono::duration<double>(std::max(0.0, arrival(k + 1) - tn)));
          continue;
        }
      } else {  // in order: wait until the tracker took the previous frame
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return stop_ || readyIdx_ <= taken_; });
        if (stop_) return;
        k = loaded_ + 1;
        if (k >= n_) break;
      }
      cv::Mat im = load_(k);
      std::lock_guard<std::mutex> l(m_);
      loaded_ = k;
      ready_ = im;
      readyIdx_ = k;
      cv_.notify_all();
      if (!camera_ && k + 1 >= n_) break;
    }
    std::lock_guard<std::mutex> l(m_);
    done_ = true;
    cv_.notify_all();
  }

  std::function<cv::Mat(int)> load_;
  const vector<double>& t_;
  const int n_;
  const double rate_;
  const bool camera_;
  const std::chrono::steady_clock::time_point wall0_;
  std::thread th_;
  std::mutex m_;
  std::condition_variable cv_;
  cv::Mat ready_;
  int readyIdx_ = -1, taken_ = -1, loaded_ = -1;
  bool done_ = false, stop_ = false;
};

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
  bool realtime = false;  // pace frames at the dataset rate, as ORB-SLAM3's own examples do
  double rateScale = 1.0; // with --realtime / --live: frame period multiplier (2 = play at half speed)
  bool live = false;      // phone-like pacing: frames arrive at the dataset rate (x rate-scale) and the tracker always
                          // takes the newest one, so frames that arrive while it is busy are dropped (as a live camera
                          // in the browser). rate-scale < 1 emulates a slower device.
  bool rtWall = false;    // every frame on a wall-clock schedule, next frame loaded in the background (honest real time)
  int maxFrames = 0;      // 0 = all frames
  bool syncMapping = false; // web port diagnostic: wait for local mapping to go idle after every frame
  bool gyro = false;        // mono: also pass the dataset IMU (only the gyro prior uses it: Web.Gyro.Prior in the settings)
  bool trace = false;       // diagnostic: per-frame trace (state, tracked points, live Twc) -> trace.txt
  int seed = -1;  // -1: ORB-SLAM3 default (TwoViewReconstruction seeds with 0)
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
    else if (a == "--seed") seed = std::stoi(next());
    else if (a == "--realtime") realtime = true;
    else if (a == "--live") live = true;
    else if (a == "--rt-wall") rtWall = true;
    else if (a == "--rate-scale") rateScale = std::stod(next());
    else if (a == "--max-frames") maxFrames = std::stoi(next());
    else if (a == "--sync-mapping") syncMapping = true;
    else if (a == "--gyro") gyro = true;
    else if (a == "--trace") trace = true;
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }
  if (dataset.empty() || mode.empty() || seq.empty() || vocab.empty() || settings.empty() || out.empty()) {
    std::cerr << "usage: orbslam3_run --dataset kitti|tumvie|euroc|tumvi --mode mono|mono_inertial --seq DIR --vocab V --settings S --out DIR [--clahe]\n";
    return 2;
  }
  const bool inertial = (mode == "mono_inertial");
  if (inertial && dataset == "kitti") { std::cerr << "KITTI sync OXTS (~10 Hz) is not usable for mono-inertial\n"; return 2; }

  // RANSAC seed for ORB-SLAM3's random sampling. Seeding once here makes TwoViewReconstruction's own
  // SeedRandOnce(0) a no-op, so different seeds give independent runs (monocular init is seed-sensitive).
  if (seed >= 0) DUtils::Random::SeedRandOnce(seed);

  Data d = (dataset == "kitti") ? loadKitti(seq)
         : (dataset == "tumvie") ? loadTumvie(seq)
         : loadAsl(seq);  // "euroc", "tumvi" (EuRoC-format export), "asl"
  int N = (int)std::min(d.images.size(), d.t.size());
  if (maxFrames > 0) N = std::min(N, maxFrames);
  fs::create_directories(out);
  out = fs::absolute(out).string();  // the working directory changes before Shutdown (see below)

  ORB_SLAM3::System SLAM(vocab, settings,
                         inertial ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::MONOCULAR,
                         /*bUseViewer=*/false);
  const float imageScale = SLAM.GetImageScale();
  cv::Ptr<cv::CLAHE> claheOp = cv::createCLAHE(3.0, cv::Size(8, 8));  // as in ORB-SLAM3's TUM-VI examples

  // First IMU sample considered: the last one at or before the first frame (as in mono_inertial_tum_vi.cc)
  size_t imuIdx = 0;
  if (inertial || gyro) {
    while (imuIdx < d.imu.size() && d.imu[imuIdx].t <= d.t[0]) ++imuIdx;
    if (imuIdx > 0) --imuIdx;
  }

  std::ofstream times(fs::path(out) / "track_times.txt");
  std::ofstream tr;
  if (trace) { tr.open(fs::path(out) / "trace.txt"); tr << std::setprecision(9); }
  vector<ORB_SLAM3::IMU::Point> meas;
  auto prep = [&](int k) {  // read + CLAHE + resize (runs on the FrameSource thread with --live / --rt-wall)
    cv::Mat im = readGray(d.images[k]);
    if (im.empty()) return im;
    if (clahe) claheOp->apply(im, im);  // (a cv::CLAHE object is used by one thread at a time here)
    if (imageScale != 1.f) cv::resize(im, im, cv::Size(im.cols * imageScale, im.rows * imageScale));
    return im;
  };
  std::unique_ptr<FrameSource> src;
  if (live || rtWall) src = std::make_unique<FrameSource>(prep, d.t, N, rateScale, live);
  const auto wall0 = std::chrono::steady_clock::now();
  int processed = 0, dropped = 0, lastIdx = -1;
  for (int i = 0; i < N; ++i) {
    cv::Mat im;
    if (src) {
      const int k = src->next(im);
      if (k < 0) break;
      dropped += k - lastIdx - 1;
      i = k;
      if (rtWall) {  // wall-clock schedule: not before the frame's time (no sleep when behind)
        const double wait = src->arrival(i) - src->now();
        if (wait > 0) std::this_thread::sleep_for(std::chrono::duration<double>(wait));
      }
    } else {
      im = prep(i);
    }
    lastIdx = i;
    ++processed;
    if (im.empty()) { std::cerr << "failed to read " << d.images[i] << "\n"; return 1; }

    meas.clear();
    if ((inertial || gyro) && i > 0) {
      while (imuIdx < d.imu.size() && d.imu[imuIdx].t <= d.t[i]) {
        const auto& m = d.imu[imuIdx++];
        meas.emplace_back(m.ax, m.ay, m.az, m.gx, m.gy, m.gz, m.t);
      }
    }
    auto t0 = std::chrono::steady_clock::now();
    const Sophus::SE3f Tcw = SLAM.TrackMonocular(im, d.t[i], meas);
    auto t1 = std::chrono::steady_clock::now();
    const double ttrack = std::chrono::duration<double>(t1 - t0).count();
#ifdef ORBSLAM3_WEB
    if (syncMapping) SLAM.WaitForMappingIdle();
#endif
    if (trace) {
      int nTracked = 0;
      for (auto* mp : SLAM.GetTrackedMapPoints()) if (mp) ++nTracked;
      const Sophus::SE3f Twc = Tcw.inverse();
      const Eigen::Quaternionf q = Twc.unit_quaternion();
      tr << i << " " << SLAM.GetTrackingState() << " " << nTracked << " " << Twc.translation().transpose() << " "
         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    times << i << " " << ttrack * 1000.0 << " " << SLAM.GetTrackingState() << "\n";
    // Real-time protocol of ORB-SLAM3's examples: wait for the rest of the frame period, which leaves time for
    // the local-mapping / loop-closing threads. Without it, results depend on the relative thread speeds.
    if (!src && realtime) {
      const double T = rateScale * ((i + 1 < N) ? d.t[i + 1] - d.t[i] : (i > 0 ? d.t[i] - d.t[i - 1] : 0.0));
      if (ttrack < T) std::this_thread::sleep_for(std::chrono::duration<double>(T - ttrack));
    }
    if (i % 200 == 0) {  // progress + map growth curve
      const auto ms = mapSize(SLAM);
      std::cerr << "frame " << i << "/" << N - 1 << " state=" << SLAM.GetTrackingState() << " map_points=" << ms[0]
                << " keyframes=" << ms[1] << " maps=" << ms[2] << " mem_mb=" << memoryMB() << "\n";
    }
  }

  const double wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  const double seqS = lastIdx > 0 ? d.t[lastIdx] - d.t[0] : 0.0;
  src.reset();
  const auto msEnd = mapSize(SLAM);  // before the final global optimisation / shutdown
  fs::current_path(out);  // REGISTER_TIMES builds write their *TimeStats.txt files to the working directory
  SLAM.Shutdown();
  const fs::path rawF = fs::path(out) / "f_euroc_ns.txt", rawKF = fs::path(out) / "kf_euroc_ns.txt";
  bool anyKF = false;  // ORB-SLAM3's savers crash when no map has a keyframe (a run ending right after a reset)
  for (ORB_SLAM3::Map* m : SLAM.WebAtlas()->GetAllMaps()) anyKF = anyKF || !m->GetAllKeyFrames().empty();
  if (anyKF) {
    SLAM.SaveTrajectoryEuRoC(rawF.string());
    SLAM.SaveKeyFrameTrajectoryEuRoC(rawKF.string());
  } else {
    std::ofstream(rawF.string()).close();
    std::ofstream(rawKF.string()).close();
  }
  euRoCToTum(rawF, fs::path(out) / "traj_tum.txt", d.tBase);
  euRoCToTum(rawKF, fs::path(out) / "kf_traj_tum.txt", d.tBase);
  // Every map's trajectory (claude/ISSUES.md H3): SaveTrajectoryEuRoC keeps only the largest map, so a multi-map run
  // (outdoors, Magistrale) loses the maps that overlap the start/end ground truth. traj_map<id>_tum.txt per map with
  // keyframes (an empty map would crash the save, G4); eval/evaluate_multimap.py scores them.
  int savedMaps = 0;
  for (ORB_SLAM3::Map* m : SLAM.WebAtlas()->GetAllMaps()) {
    if (m->GetAllKeyFrames().empty()) continue;
    const std::string id = std::to_string(m->GetId());
    const fs::path rawM = fs::path(out) / ("map" + id + "_euroc_ns.txt");
    SLAM.SaveTrajectoryEuRoC(rawM.string(), m);
    euRoCToTum(rawM, fs::path(out) / ("traj_map" + id + "_tum.txt"), d.tBase);
    ++savedMaps;
  }

  std::ofstream info(fs::path(out) / "run_info.txt");
  info << std::setprecision(17) << "dataset " << dataset << "\nmode " << mode << "\nseq " << seq
       << "\nsettings " << settings << "\nclahe " << clahe << "\nseed " << seed << "\nrealtime " << realtime
       << "\nrate_scale " << rateScale << "\nlive " << live << "\nrt_wall " << rtWall << "\nsync_mapping " << syncMapping
       << "\ngyro " << gyro << "\nframes " << N << "\nprocessed " << processed << "\ndropped " << dropped
       << "\nwall_s " << wallS << "\nseq_s " << seqS
       << "\neffective_rate " << (wallS > 0 ? seqS / wallS : 0.0)  // 1 = real time (x 1/rate_scale when scaled)
       << "\nmap_points " << msEnd[0] << "\nkeyframes " << msEnd[1] << "\nmaps " << msEnd[2] << "\nsaved_maps " << savedMaps
       << "\nmem_mb " << memoryMB()  // WASM heap size or native peak RSS
       << "\nt_base " << d.tBase << "\n";
  return 0;
}
