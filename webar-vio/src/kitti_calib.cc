#include "kitti_calib.h"
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cctype>
#include <algorithm>

// ---------- helpers ----------
static inline std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) a++;
  while (b > a && std::isspace((unsigned char)s[b-1])) b--;
  return s.substr(a, b-a);
}

static inline std::vector<double> parseNums(const std::string& rhs) {
  std::vector<double> v;
  std::stringstream ss(rhs);
  double x;
  while (ss >> x) v.push_back(x);
  return v;
}

static inline std::unordered_map<std::string, std::vector<double>> parseKeyVals(const std::string& txt) {
  std::unordered_map<std::string, std::vector<double>> out;
  std::stringstream ss(txt);
  std::string line;
  while (std::getline(ss, line)) {
    line = trim(line);
    if (line.empty()) continue;
    auto pos = line.find(':');
    if (pos == std::string::npos) continue;
    std::string key = trim(line.substr(0, pos));
    std::string rhs = trim(line.substr(pos+1));
    auto nums = parseNums(rhs);
    if (!nums.empty()) out[key] = std::move(nums);
  }
  return out;
}

static inline cv::Matx44d makeT_from_Rt(const std::vector<double>& R9, const std::vector<double>& t3) {
  cv::Matx44d T = cv::Matx44d::eye();
  if (R9.size() == 9) {
    T(0,0)=R9[0]; T(0,1)=R9[1]; T(0,2)=R9[2];
    T(1,0)=R9[3]; T(1,1)=R9[4]; T(1,2)=R9[5];
    T(2,0)=R9[6]; T(2,1)=R9[7]; T(2,2)=R9[8];
  }
  if (t3.size() >= 3) {
    T(0,3)=t3[0]; T(1,3)=t3[1]; T(2,3)=t3[2];
  }
  return T;
}

static inline cv::Matx33d mat33_from44(const cv::Matx44d& T) {
  return cv::Matx33d(
    T(0,0),T(0,1),T(0,2),
    T(1,0),T(1,1),T(1,2),
    T(2,0),T(2,1),T(2,2)
  );
}
static inline cv::Vec3d vec3_from44(const cv::Matx44d& T) {
  return cv::Vec3d(T(0,3), T(1,3), T(2,3));
}

bool parseKittiCalibFromTexts_Image02(
  const std::string& cam2cam_txt,
  const std::string& velo2cam_txt,
  const std::string& imu2velo_txt,
  KittiCalibOut& out
) {
  // ---- cam intrinsics from calib_cam_to_cam: use P_rect_02 (3x4) ----
  auto camKV = parseKeyVals(cam2cam_txt);
  auto itP = camKV.find("P_rect_02");
  if (itP == camKV.end() || itP->second.size() != 12) return false;

  const auto& P = itP->second;
  // P = [fx 0 cx Tx; 0 fy cy Ty; 0 0 1 0] typically
  const double fx = P[0];
  const double fy = P[5];
  const double cx = P[2];
  const double cy = P[6];

  out.K = cv::Matx33d(
    fx, 0,  cx,
    0,  fy, cy,
    0,  0,  1
  );

  // ---- extrinsics: camera-from-imu = (velo->cam) * (imu->velo) ----
  // velo_to_cam file usually provides "R" and "T" for Tr_velo_to_cam or similar.
  // imu_to_velo provides "R" and "T" for Tr_imu_to_velo or similar.
  auto veloKV = parseKeyVals(velo2cam_txt);
  auto imuKV  = parseKeyVals(imu2velo_txt);

  // robustly locate keys for R/T:
  // common: velo_to_cam: "R" "T" OR "R:" "T:" (we already stripped ':')
  // some KITTI files use "R:" "T:" with the same names "R" and "T".
  auto findRT = [](const std::unordered_map<std::string,std::vector<double>>& kv,
                   std::vector<double>& R9, std::vector<double>& t3) -> bool {
    // direct
    auto iR = kv.find("R");
    auto iT = kv.find("T");
    if (iR != kv.end() && iT != kv.end() && iR->second.size()==9 && iT->second.size()>=3) {
      R9 = iR->second;
      t3 = iT->second;
      return true;
    }
    // KITTI sometimes uses "Tr" in other files, but for these 2 it’s usually R/T.
    // Try fallbacks:
    iR = kv.find("R_imu2velo");
    iT = kv.find("T_imu2velo");
    if (iR != kv.end() && iT != kv.end()) { R9=iR->second; t3=iT->second; return true; }

    iR = kv.find("R_velo2cam");
    iT = kv.find("T_velo2cam");
    if (iR != kv.end() && iT != kv.end()) { R9=iR->second; t3=iT->second; return true; }

    // Try "Tr" 3x4 flattened
    auto iTr = kv.find("Tr");
    if (iTr != kv.end() && iTr->second.size()==12) {
      const auto& A = iTr->second;
      R9 = {A[0],A[1],A[2], A[4],A[5],A[6], A[8],A[9],A[10]};
      t3 = {A[3],A[7],A[11]};
      return true;
    }
    return false;
  };

  std::vector<double> Rv9, tv3, Ri9, ti3;
  if (!findRT(veloKV, Rv9, tv3)) return false;
  if (!findRT(imuKV,  Ri9, ti3)) return false;

  const cv::Matx44d T_cam_from_velo = makeT_from_Rt(Rv9, tv3);
  const cv::Matx44d T_velo_from_imu = makeT_from_Rt(Ri9, ti3);
  const cv::Matx44d T_cam_from_imu  = T_cam_from_velo * T_velo_from_imu;

  out.R_ci = mat33_from44(T_cam_from_imu);
  out.t_ci = vec3_from44(T_cam_from_imu);
  return true;
}
