// Headless native runner for the webar-vio System.
//
// Feeds a dataset sequence into System exactly the way the browser runners do
// (kitti.js / tumvi.js): same timestamp parsing and rebasing, same IMU windowing,
// same RGBA->gray integer formula, same call order. This makes native runs usable
// for accuracy evaluation; browser runs remain the source for timing numbers.
//
// Usage:
//   vio_run --dataset kitti|tumvie --seq <sequence dir> --out <output dir>
//           [--skip N] [--max-frames N]
//
// Outputs (in --out):
//   traj_tum.txt        "t tx ty tz qx qy qz qw" — raw System camera pose Twc (camera frame, OpenCV axes),
//                       t = absolute dataset time in seconds. No axis permutation.
//   legacy_poses.csv    identical format/content to the browser's "Export poses CSV" (incl. its axis remap),
//                       for reproducing the original evaluation scripts.
//   perf.csv            same columns as the browser's performance CSV (native timings!).
//   run_info.txt        arguments, frame counts, calibration used.

#include "system.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct ImuSample { double t, ax, ay, az, wx, wy, wz; };

struct Frame { fs::path path; double t; int idx; };  // idx = dataset frame index (from the file name)

struct Sequence {
  std::vector<Frame> frames;       // only frames whose image exists, in index order
  std::vector<ImuSample> imu;      // rebased seconds
  double tBase = 0.0;              // absolute time of rebased t=0 (seconds)
  int missingImages = 0;
};

// Dataset files are named by frame index (KITTI 0000000123.png, TUM-VIE 00123.jpg). Pair files with
// timestamps by that index, never by list position: a single missing file would otherwise shift every
// later timestamp.
static int fileIndex(const fs::path& p) {
  const std::string s = p.stem().string();
  std::string digits;
  for (char c : s) if (std::isdigit((unsigned char)c)) digits.push_back(c);
  return digits.empty() ? -1 : std::stoi(digits);
}

static std::vector<Frame> pairByIndex(const std::vector<fs::path>& files, const std::vector<double>& ts,
                                      int& missing) {
  std::vector<Frame> out;
  for (const auto& f : files) {
    const int k = fileIndex(f);
    if (k >= 0 && k < (int)ts.size()) out.push_back({f, ts[k], k});
  }
  missing = (int)ts.size() - (int)out.size();
  return out;
}

// ---------------------------------------------------------------- helpers

static bool naturalLess(const std::string& a, const std::string& b) {
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (std::isdigit((unsigned char)a[i]) && std::isdigit((unsigned char)b[j])) {
      size_t i2 = i, j2 = j;
      while (i2 < a.size() && std::isdigit((unsigned char)a[i2])) ++i2;
      while (j2 < b.size() && std::isdigit((unsigned char)b[j2])) ++j2;
      std::string na = a.substr(i, i2 - i), nb = b.substr(j, j2 - j);
      na.erase(0, std::min(na.find_first_not_of('0'), na.size()));
      nb.erase(0, std::min(nb.find_first_not_of('0'), nb.size()));
      if (na.size() != nb.size()) return na.size() < nb.size();
      if (na != nb) return na < nb;
      i = i2; j = j2;
    } else {
      char ca = (char)std::tolower((unsigned char)a[i]), cb = (char)std::tolower((unsigned char)b[j]);
      if (ca != cb) return ca < cb;
      ++i; ++j;
    }
  }
  return a.size() < b.size();
}

static std::vector<fs::path> listImages(const fs::path& dir) {
  std::vector<fs::path> out;
  if (!fs::is_directory(dir)) return out;
  for (const auto& e : fs::directory_iterator(dir)) {
    if (!e.is_regular_file()) continue;
    std::string ext = e.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") out.push_back(e.path());
  }
  std::sort(out.begin(), out.end(), [](const fs::path& a, const fs::path& b) {
    return naturalLess(a.filename().string(), b.filename().string());
  });
  return out;
}

static std::string readFile(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.string());
  std::stringstream ss; ss << f.rdbuf();
  return ss.str();
}

static std::vector<std::string> splitLines(const std::string& text) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string line;
  while (std::getline(ss, line)) {
    size_t a = line.find_first_not_of(" \t\r");
    if (a == std::string::npos) continue;
    size_t b = line.find_last_not_of(" \t\r");
    out.push_back(line.substr(a, b - a + 1));
  }
  return out;
}

static bool parseNumbers(const std::string& line, std::vector<double>& v) {
  v.clear();
  std::stringstream ss(line);
  std::string tok;
  while (ss >> tok) {
    char* end = nullptr;
    double x = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0' || !std::isfinite(x)) return false;
    v.push_back(x);
  }
  return true;
}

// Rotation matrix (row-major) -> quaternion (x,y,z,w)
static void rotToQuat(const std::array<double, 9>& R, double q[4]) {
  const double m00 = R[0], m01 = R[1], m02 = R[2];
  const double m10 = R[3], m11 = R[4], m12 = R[5];
  const double m20 = R[6], m21 = R[7], m22 = R[8];
  const double tr = m00 + m11 + m22;
  double x, y, z, w;
  if (tr > 0) {
    double S = std::sqrt(tr + 1.0) * 2;
    w = 0.25 * S; x = (m21 - m12) / S; y = (m02 - m20) / S; z = (m10 - m01) / S;
  } else if (m00 > m11 && m00 > m22) {
    double S = std::sqrt(1.0 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / S; x = 0.25 * S; y = (m01 + m10) / S; z = (m02 + m20) / S;
  } else if (m11 > m22) {
    double S = std::sqrt(1.0 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / S; x = (m01 + m10) / S; y = 0.25 * S; z = (m12 + m21) / S;
  } else {
    double S = std::sqrt(1.0 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / S; x = (m02 + m20) / S; y = (m12 + m21) / S; z = 0.25 * S;
  }
  const double n = std::sqrt(x * x + y * y + z * z + w * w);
  q[0] = x / n; q[1] = y / n; q[2] = z / n; q[3] = w / n;
}

// ---------------------------------------------------------------- KITTI raw (mirrors kitti.js)

// "2011-09-26 13:02:25.964389445" -> seconds. kitti.js uses local-time Date for the day; only
// differences matter, so UTC (timegm) gives identical relative timestamps.
static bool parseKittiTimestamp(const std::string& s, double& out) {
  int Y, M, D, hh, mm, ss;
  char frac[32] = {0};
  if (std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d.%31[0-9]", &Y, &M, &D, &hh, &mm, &ss, frac) < 6) return false;
  std::tm tm{}; tm.tm_year = Y - 1900; tm.tm_mon = M - 1; tm.tm_mday = D;
  const double day = (double)timegm(&tm);
  std::string f(frac);
  f = f.substr(0, 9);
  while (f.size() < 9) f.push_back('0');
  out = day + hh * 3600.0 + mm * 60.0 + ss + std::stod("0." + f);
  return true;
}

static std::vector<double> loadKittiTimestamps(const fs::path& p) {
  std::vector<double> out;
  for (const auto& l : splitLines(readFile(p))) {
    double t;
    if (parseKittiTimestamp(l, t)) out.push_back(t);
  }
  return out;
}

static fs::path findUp(const fs::path& seq, const std::string& name) {
  for (fs::path d : {seq, seq.parent_path()}) {
    if (fs::exists(d / name)) return d / name;
  }
  return {};
}

static Sequence loadKitti(const fs::path& seq, System& sys, std::ostream& info) {
  Sequence S;
  const std::vector<fs::path> images = listImages(seq / "image_02" / "data");
  if (images.empty()) throw std::runtime_error("no images in image_02/data");

  const fs::path tsImg = seq / "image_02" / "timestamps.txt";
  const fs::path tsOxts = seq / "oxts" / "timestamps.txt";
  if (!fs::exists(tsImg) || !fs::exists(tsOxts))
    throw std::runtime_error("need image_02/timestamps.txt and oxts/timestamps.txt");

  std::vector<double> imageTS = loadKittiTimestamps(tsImg);
  std::vector<double> oxtsTS = loadKittiTimestamps(tsOxts);
  // kitti.js: rebase BOTH streams to the same t0 = min(first image ts, first oxts ts)
  S.tBase = std::min(imageTS.front(), oxtsTS.front());
  for (auto& t : imageTS) t -= S.tBase;
  for (auto& t : oxtsTS) t -= S.tBase;

  // OXTS: first line of each file. kitti.js uses the vehicle-frame channels:
  //   accel = af, al, au (idx 14,15,16); gyro = wf, wl, wu (idx 20,21,22)
  std::vector<fs::path> oxtsFiles;
  for (const auto& e : fs::directory_iterator(seq / "oxts" / "data"))
    if (e.is_regular_file() && e.path().extension() == ".txt") oxtsFiles.push_back(e.path());
  std::sort(oxtsFiles.begin(), oxtsFiles.end(), [](const fs::path& a, const fs::path& b) {
    return naturalLess(a.filename().string(), b.filename().string());
  });

  std::vector<double> v;
  int missingOxts = (int)oxtsTS.size();
  for (const auto& f : oxtsFiles) {
    const int k = fileIndex(f);
    if (k < 0 || k >= (int)oxtsTS.size()) continue;
    auto lines = splitLines(readFile(f));
    if (lines.empty() || !parseNumbers(lines[0], v) || v.size() < 23) continue;  // kitti.js: null -> skipped
    S.imu.push_back({oxtsTS[k], v[14], v[15], v[16], v[20], v[21], v[22]});
    --missingOxts;
  }

  S.frames = pairByIndex(images, imageTS, S.missingImages);
  info << "missing_images " << S.missingImages << "\nmissing_oxts " << missingOxts << "\n";

  // Calibration (kitti.js passes the three raw text files into WASM before initSystem)
  const fs::path c2c = findUp(seq, "calib_cam_to_cam.txt");
  const fs::path v2c = findUp(seq, "calib_velo_to_cam.txt");
  const fs::path i2v = findUp(seq, "calib_imu_to_velo.txt");
  bool calibOk = false;
  if (!c2c.empty() && !v2c.empty() && !i2v.empty())
    calibOk = sys.setKittiCalibFromTexts(readFile(c2c), readFile(v2c), readFile(i2v));
  info << "kitti_calib_applied " << (calibOk ? "yes" : "no") << "\n";
  if (!calibOk) throw std::runtime_error("KITTI calibration files not found / not parsed");

  // Remaining kitti.js init sequence
  cv::Mat first = cv::imread(S.frames[0].path.string(), cv::IMREAD_UNCHANGED);
  const int W = first.cols, H = first.rows;
  const double FOVY = 45.0;
  const double fy = H / (2 * std::tan((FOVY * M_PI / 180) / 2));
  const double fx = fy * ((double)W / H);
  sys.setUseFisheye(false);
  sys.setAccelIsSpecificForce(true);
  sys.init(W, H, fx, fy, W * 0.5, H * 0.5);  // intrinsics overridden by KITTI calib inside init()
  info << "image_size " << W << " " << H << "\n";
  return S;
}

// ---------------------------------------------------------------- TUM-VIE (mirrors tumvi.js + kitti.js TUM branch)

static std::vector<double> parseUsFloatLines(const std::string& text) {
  std::vector<double> out;
  for (const auto& s : splitLines(text)) {
    if (s[0] == '#') continue;
    char* end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end != s.c_str() && std::isfinite(v)) out.push_back(v);
  }
  return out;
}

static Sequence loadTumvie(const fs::path& seq, System& sys, std::ostream& info) {
  Sequence S;
  const std::vector<fs::path> images = listImages(seq / "left_images");
  if (images.empty()) throw std::runtime_error("no images in left_images");

  // camera-calibration.json is {"value0": {...}} (Basalt format)
  cv::FileStorage fsCal((seq / "camera-calibration.json").string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (!fsCal.isOpened()) throw std::runtime_error("cannot read camera-calibration.json");
  cv::FileNode root = fsCal["value0"].empty() ? fsCal.root() : fsCal["value0"];

  double camOffset = 0.0;
  if (!root["cam_time_offset_ns"].empty()) camOffset = (double)root["cam_time_offset_ns"] * 1e-9;

  std::vector<double> tsUs = parseUsFloatLines(readFile(seq / "left_images" / "image_timestamps_left.txt"));
  std::vector<double> imageTS;
  for (double x : tsUs) imageTS.push_back(x * 1e-6 + camOffset);
  S.tBase = imageTS.empty() ? 0.0 : imageTS[0];
  for (auto& t : imageTS) t -= S.tBase;

  // IMU: "t_us gx gy gz ax ay az" -> rebased to the image t0
  std::vector<double> v;
  for (const auto& s : splitLines(readFile(seq / "imu_data.txt"))) {
    if (s[0] == '#') continue;
    if (!parseNumbers(s, v) || v.size() < 7) continue;
    S.imu.push_back({v[0] * 1e-6 - S.tBase, v[4], v[5], v[6], v[1], v[2], v[3]});
  }

  S.frames = pairByIndex(images, imageTS, S.missingImages);
  info << "missing_images " << S.missingImages << "\n";

  // Extrinsics: tumvi.js calls setImuToCamQuat with T_imu_cam[0] as-is
  cv::FileNode T = root["T_imu_cam"][0];
  sys.setImuToCamQuat((double)T["qx"], (double)T["qy"], (double)T["qz"], (double)T["qw"],
                      (double)T["px"], (double)T["py"], (double)T["pz"]);

  cv::FileNode cam = root["intrinsics"][0];
  cv::FileNode intr = cam["intrinsics"];
  const double fx = intr["fx"], fy = intr["fy"], cx = intr["cx"], cy = intr["cy"];
  std::string camType = (std::string)cam["camera_type"];
  std::transform(camType.begin(), camType.end(), camType.begin(), ::tolower);

  sys.setUseFisheye(false);
  if ((camType.find("kb4") != std::string::npos || camType.find("fisheye") != std::string::npos) &&
      !intr["k1"].empty() && !intr["k4"].empty()) {
    sys.setKb4Distortion(intr["k1"], intr["k2"], intr["k3"], intr["k4"]);
    info << "kb4 " << (double)intr["k1"] << " " << (double)intr["k2"] << " "
         << (double)intr["k3"] << " " << (double)intr["k4"] << "\n";
  }
  sys.setAccelIsSpecificForce(true);

  cv::Mat first = cv::imread(S.frames[0].path.string(), cv::IMREAD_UNCHANGED);
  sys.init(first.cols, first.rows, fx, fy, cx, cy);
  info << "image_size " << first.cols << " " << first.rows << "\n"
       << "intrinsics " << fx << " " << fy << " " << cx << " " << cy << "\n"
       << "cam_time_offset_s " << camOffset << "\n";
  return S;
}

// ---------------------------------------------------------------- image -> gray, exactly like the browser path

// Browser: canvas RGBA -> gray[j] = (77*R + 150*G + 29*B) >> 8. A grayscale image decodes to R=G=B=v,
// for which the formula returns v exactly.
static void toBrowserGray(const cv::Mat& img, cv::Mat& gray) {
  if (img.channels() == 1) { img.copyTo(gray); return; }
  gray.create(img.rows, img.cols, CV_8UC1);
  const int ch = img.channels();  // BGR or BGRA
  for (int r = 0; r < img.rows; ++r) {
    const uint8_t* p = img.ptr<uint8_t>(r);
    uint8_t* g = gray.ptr<uint8_t>(r);
    for (int c = 0; c < img.cols; ++c, p += ch) {
      const int B = p[0], G = p[1], R = p[2];
      g[c] = (uint8_t)((77 * R + 150 * G + 29 * B) >> 8);
    }
  }
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
  std::string dataset, seqDir, outDir;
  int skip = 0, maxFrames = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(2); }
      return argv[++i];
    };
    if (a == "--dataset") dataset = next();
    else if (a == "--seq") seqDir = next();
    else if (a == "--out") outDir = next();
    else if (a == "--skip") skip = std::stoi(next());
    else if (a == "--max-frames") maxFrames = std::stoi(next());
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }
  if ((dataset != "kitti" && dataset != "tumvie") || seqDir.empty() || outDir.empty()) {
    std::cerr << "usage: vio_run --dataset kitti|tumvie --seq <dir> --out <dir> [--skip N] [--max-frames N]\n";
    return 2;
  }
  fs::create_directories(outDir);
  std::ofstream info(fs::path(outDir) / "run_info.txt");
  info << std::setprecision(17)
       << "dataset " << dataset << "\nseq " << seqDir << "\nopencv " << CV_VERSION << "\n";

  System sys;
  sys.setTrackerType(0);  // browser runner forces KLT
  Sequence S;
  try {
    S = (dataset == "kitti") ? loadKitti(seqDir, sys, info) : loadTumvie(seqDir, sys, info);
  } catch (const std::exception& e) {
    std::cerr << "load failed: " << e.what() << "\n";
    return 1;
  }
  const int N = (int)S.frames.size();
  if (S.missingImages > 0)
    std::cerr << "WARNING: " << S.missingImages << " frames have no image file; they are skipped\n";
  info << "frames " << N << "\nimu_samples " << S.imu.size() << "\nt_base " << S.tBase << "\n";

  std::ofstream tum(fs::path(outDir) / "traj_tum.txt");
  std::ofstream legacy(fs::path(outDir) / "legacy_poses.csv");
  std::ofstream perf(fs::path(outDir) / "perf.csv");
  tum << std::setprecision(17);
  legacy << std::scientific << std::setprecision(16);
  perf << std::scientific << std::setprecision(16);
  tum << "# t tx ty tz qx qy qz qw  (Twc: camera-to-world, camera frame = OpenCV x-right y-down z-forward)\n";
  legacy << "frame,t,x,y,z,yaw_rad,pitch_rad,roll_rad\n";
  perf << "frame,t,num_kfs,num_mps,num_keypoints,wasm_klt_ms,wasm_total_ms,wasm_imu_ms,wasm_seed_ms,imu_hz\n";

  const int startIdx = std::min(N - 1, std::max(0, skip));
  const int limit = (maxFrames > 0) ? std::min(N, startIdx + maxFrames) : N;
  size_t imuIdx = 0;
  cv::Mat img, gray;

  for (int i = startIdx; i < limit; ++i) {
    const Frame& F = S.frames[i];
    const double tNow = F.t;

    img = cv::imread(F.path.string(), cv::IMREAD_UNCHANGED);
    if (img.empty()) { std::cerr << "failed to read " << F.path << "\n"; return 1; }
    if (img.depth() != CV_8U) img.convertTo(img, CV_8U);
    toBrowserGray(img, gray);
    if (!gray.isContinuous()) gray = gray.clone();

    // kitti.js feedImuWindow(tPrev, tNow): skip samples < tPrev, feed samples <= tNow (persistent index)
    const double tPrev = (i > startIdx) ? S.frames[i - 1].t : (tNow - 0.1);
    while (imuIdx < S.imu.size() && S.imu[imuIdx].t < tPrev) ++imuIdx;
    while (imuIdx < S.imu.size() && S.imu[imuIdx].t <= tNow) {
      const auto& m = S.imu[imuIdx++];
      sys.feedImu(m.t, m.ax, m.ay, m.az, m.wx, m.wy, m.wz);
    }

    sys.feedFrame(gray.data, tNow, gray.cols, gray.rows, false);

    const auto twc = sys.getTwc();
    const auto ypr = sys.getYPR();
    const auto R = sys.getRwc();
    double q[4];
    rotToQuat(R, q);

    tum << (tNow + S.tBase) << " " << twc[0] << " " << twc[1] << " " << twc[2] << " "
        << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << "\n";
    // Browser export applies x=twc[1], y=-twc[2], z=twc[0] (kept only to reproduce old numbers)
    legacy << F.idx << "," << tNow << "," << twc[1] << "," << -twc[2] << "," << twc[0] << ","
           << ypr[0] << "," << ypr[1] << "," << ypr[2] << "\n";
    perf << F.idx << "," << tNow << "," << (double)sys.getNumKFs() << "," << (double)sys.getNumMPs() << ","
         << (double)sys.getNumKeypoints() << "," << sys.getLastKltMS() << "," << sys.getLastTotalMS() << ","
         << sys.getLastImuMS() << "," << sys.getLastSeedMS() << "," << sys.getImuHz() << "\n";

    if ((i - startIdx) % 200 == 0)
      std::cerr << "frame " << i << "/" << limit - 1 << "  kfs=" << sys.getNumKFs()
                << " mps=" << sys.getNumMPs() << "\n";
  }
  info << "frames_processed " << (limit - startIdx) << "\n";
  std::cerr << "done: " << (limit - startIdx) << " frames -> " << outDir << "\n";
  return 0;
}
