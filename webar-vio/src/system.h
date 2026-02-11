#pragma once

#include <vector>
#include <cstdint>
#include <array>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include "kitti_calib.h"
#include "vio_backend.h"
#include "vio_init.h"
#include "imu_preint.h"
#include "vio_initializer.h"

class System {
public:
  System();

  // Initialize image size and camera intrinsics
  void init(int width, int height, double fx, double fy, double cx, double cy);

  // img: RGBA or Gray buffer; ts in seconds
  // stride is not required; buffer is assumed tightly packed per row.
  void feedFrame(const uint8_t* img, double ts, int width, int height, bool isRGBA);

  // Flattened [x0,y0, x1,y1, ...] in FULL-RES pixel coords (origin top-left)
  std::vector<double> getPoints2D() const;

  int getNumKeypoints() const { return static_cast<int>(ptsCur_.size()); }
  int getTrackState()   const { return trackingState_; } // 0=uninit,1=tracking
  double getLastTS()    const { return lastTS_; }
  double getLastTotalMS() const { return t_last_total_ms_; }
  double getLastKltMS()   const { return t_last_klt_ms_; }
  double getLastSeedMS()  const { return t_last_seed_ms_; }
  int maxReturnPts_ = 200;
  double getLastMeanY() const { return lastMeanY_; }
  std::array<int,2> getLastProcWH() const { return { curProc_.cols, curProc_.rows }; }
  enum class TrackerType { KLT = 0, ORB = 1, HYBRID = 2 };

  void setTrackerType(int t) {
    if      (t == 2) trackerType_ = TrackerType::HYBRID;
    else if (t == 1) trackerType_ = TrackerType::ORB;
    else             trackerType_ = TrackerType::KLT;
  }
  int getTrackerType() const {
    switch (trackerType_) {
      case TrackerType::HYBRID: return 2;
      case TrackerType::ORB:    return 1;
      default:                  return 0;
    }
  }  

  // Pass raw KITTI calib text files (loaded in JS) into WASM
  bool setKittiCalibFromTexts(const std::string& cam2cam,
                              const std::string& velo2cam,
                              const std::string& imu2velo);

  double getLastOrbMS() const { return t_last_orb_ms_; }
  void setHybridEveryN(int n) { hybridEveryN_ = std::max(1, n); }
  int  getHybridEveryN() const { return hybridEveryN_; }
  bool getRanOrbThisFrame() const { return ranOrbThisFrame_; }
  int  getOrbKFCount() const { return orbKFCount_; }
  uint64_t getHybFrameIdx() const { return hybFrameIdx_; }
  // --- VO / path telemetry ---
  // Returns a flattened [x0,z0, x1,z1, ...] in arbitrary scale (monocular)
  std::vector<float> getPathXZ() const;
  // --- Pose accessors for UI ---
  // World position (x,y,z)
  std::array<double,3> getTwc() const {
    return { twc_[0], twc_[1], twc_[2] };
  }
  // Heading (yaw on XZ), pitch, roll in radians (right-handed, OpenCV: x right, y down, z forward)
  // yaw: angle of forward (Rwc_.col(2)) projected on XZ
  // pitch: elevation of forward
  // roll: rotation around forward axis using right/up
  std::array<double,3> getYPR() const;
  // --- E/H gate telemetry (for HUD/logging) ---
  // model: 0=NONE, 1=E, 2=H
  int    getEHModel()       const { return ehModel_; }
  int    getEHInliersE()    const { return ehInliersE_; }
  int    getEHInliersH()    const { return ehInliersH_; }
  double getEHParallaxDeg() const { return ehParallaxDeg_; }
  // Mapping stats (public getters)
  int getNumKFs() const { return (int)kfs_.size(); }
  int getNumMPs() const { return (int)mps_.size(); }
  double getLastImuMS() const { return t_last_imu_ms_; }
  // IMU sample in SI units, timestamp in seconds (same clock domain as feedFrame ts)
  void feedImu(double ts,
              double ax, double ay, double az,    // m/s^2
              double gx, double gy, double gz);   // rad/s
  int    getImuUsedThisFrame() const { return imuUsedThisFrame_; }
  double getImuHz() const { return imuHz_; }
  int    getImuSamplesUsedThisFrame() const { return imuSamplesUsedThisFrame_; }
  int    getImuBufSize() const { return (int)imuBuf_.size(); }
  // --- IMU (gyro-only) delta rotation debug (between last frame ts and this frame ts) ---
  std::array<double,3> getImuDeltaYPR() const { return { imuDeltaYPR_[0], imuDeltaYPR_[1], imuDeltaYPR_[2] }; }
  std::array<double,3> getImuDeltaRodrigues() const { return { imuDeltaRod_[0], imuDeltaRod_[1], imuDeltaRod_[2] }; }
  double getImuDeltaAngleDeg() const { return imuDeltaAngleDeg_; }
  
  void setImuToCamQuat(double qx, double qy, double qz, double qw,
                         double px, double py, double pz);

  void setKb4Distortion(double k1,double k2,double k3,double k4) {
    D_kb4_ = cv::Vec4d(k1,k2,k3,k4);
    useFisheye_ = true;
  }
  void setUseFisheye(bool on) { useFisheye_ = on; }

private:
  int   procScale_      = 1;        // 2 => process at half-res (major speedup)
  int   kltWin_         = 25;
  int   kltLevels_      = 4;
  float kltErrMax_      = 6.f;     // LK per-point error gate
  float fbMax_          = 1.6f;     // forward-backward gate (pixels)
  int   cellSize_       = 8;       // grid cell size for seeding (processing scale) ***** Scale DOWN 
  int   targetKps_      = 1200;      // feature budget at processing scale ***** Scale UP
  int   descEveryN_     = 0;        // ORB compute cadence (frames); 0 disables
  int   maxTracks_    =200;  // hard ceiling after tracking+reseeding
  double t_last_total_ms_ = 0.0;
  double t_last_klt_ms_   = 0.0;
  double t_last_seed_ms_  = 0.0;
  double lastMeanY_ = -1.0;
  bool imuHadDeltaThisFrame_ = false;
  // --- per-frame gyro delta debug ---
  cv::Vec3d imuDeltaYPR_ = cv::Vec3d(0,0,0);      // radians (delta yaw/pitch/roll)
  cv::Vec3d imuDeltaRod_ = cv::Vec3d(0,0,0);      // Rodrigues vector (axis * angle), radians
  double    imuDeltaAngleDeg_ = 0.0;              // magnitude of imuDeltaRod_ in degrees

  struct ImuState {
    cv::Matx33d Rwb = cv::Matx33d::eye();  // world-from-body (IMU body)
    cv::Vec3d   vwb = cv::Vec3d(0,0,0);    // (optional later)
    cv::Vec3d   pwb = cv::Vec3d(0,0,0);    // (optional later)
    cv::Vec3d   bg  = cv::Vec3d(0,0,0);    // gyro bias (later)
    cv::Vec3d   ba  = cv::Vec3d(0,0,0);    // accel bias (later)
  };
  ImuState imuState_;
  
  // Gravity direction estimate in world (unit vector), and magnitude
  cv::Vec3d gDirW_   = cv::Vec3d(0, 1, 0);// world: X=Up => gravity points -X (Down)
  double    gMag_  = 9.81;
  
  // Complementary filter gain (tune)
  double imuAccKp_ = 2.0;
  cv::Matx33d Rcb_ = cv::Matx33d::eye(); // camera-from-body (IMU->Cam). Calibrate later.

  // Camera-from-IMU extrinsics (KITTI provides these)
  cv::Vec3d t_ci_ = cv::Vec3d(0,0,0);   // camera-from-imu translation (meters)
  bool haveKittiCalib_ = false;
  cv::Matx33d kittiK_ = cv::Matx33d::eye();

  // ===== Tight-coupled VIO backend =====
  VioBackend vio_;
  bool vioInitDone_ = false;
  double vioLastKfTs_ = 0.0;

  VioInitializer vioInit_;
  bool vioMetricInitDone_ = false;
  double vioScale_ = 1.0;

  // Map dataset IMU axes -> your IMU-body axes (start as identity).
  // If yaw is weak/incorrect, this is where we fix axis order/signs.
  cv::Matx33d R_b_imu_ = cv::Matx33d::eye();

  // keep IMU in a backend-friendly vector
  std::vector<ImuMeas> imuMeas_;

  // last reprojection obs (built from map correspondences)
  std::vector<FeatureObs> vioLastObs_;

  // helper: push keyframe into backend + optimize + publish pose
  void vioOnKeyframe(double ts);

  cv::TermCriteria termcrit_{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01};

  // Image geometry / intrinsics
  int imgW_ = 0, imgH_ = 0;
  double fx_ = 0., fy_ = 0., cx_ = 0., cy_ = 0.;

  // --- Fisheye (KB4) support for TUM-VI ---
  bool useFisheye_ = false;
  cv::Vec4d D_kb4_ = cv::Vec4d(0,0,0,0); // k1,k2,k3,k4
  cv::Mat map1_, map2_;                  // remap tables (proc-scale)
  cv::Mat tmpProc_;                      // temp before undistort

  // Working images (reused every frame)
  cv::Mat prevGray_, curGray_;   // full-res gray
  cv::Mat prevProc_, curProc_;   // downscaled gray (working resolution)

  // Pyramids (processing scale)
  std::vector<cv::Mat> pyrPrev_, pyrCur_;

  // Tracks (processing scale coordinates)
  std::vector<cv::Point2f> ptsPrev_, ptsCur_;
  int frameCount_ = 0;

  // State
  int trackingState_ = 0; // 0=uninitialized, 1=tracking
  double lastTS_ = 0.0;

  TrackerType trackerType_ = TrackerType::KLT;

  // Hybrid config/state
  int hybridEveryN_ = 8;               // default: run ORB every 4th frame
  uint64_t hybFrameIdx_ = 0;            // increments each feedFrame
  bool     ranOrbThisFrame_ = false;
  uint64_t lastOrbKF_ = 0;       // last frame idx that ran ORB
  int      orbKFCount_ = 0;      // number of ORB keyframes run

  // ORB objects + frame-to-frame state
  cv::Ptr<cv::ORB> orb_;
  std::vector<cv::KeyPoint> orbPrevKps_, orbCurKps_;
  cv::Mat orbPrevDesc_, orbCurDesc_;

  // Tunables (reasonable defaults; tweak later)
  int   orbNFeatures_     = 600;
  float orbScaleFactor_   = 1.2f;
  int   orbNLevels_       = 5;
  int   orbEdgeThreshold_ = 31;
  int   orbFirstLevel_    = 0;
  int   orbWtaK_          = 2;
  cv::ORB::ScoreType orbScore_ = cv::ORB::HARRIS_SCORE;
  int   orbPatchSize_     = 31;
  int   orbFastThreshold_ = 12;

  // ===== Mapping state (SFM → VIO) =====
  struct Keyframe {
    int id = -1;
    cv::Matx33d Rwc;   // world-from-camera
    cv::Vec3d   twc;
    std::vector<cv::KeyPoint> kps; // ORB keypoints at processing scale
    cv::Mat desc;                   // ORB descriptors (rows=kps)
  };
  struct MapPoint {
    cv::Vec3d Xw;
    cv::Mat   desc;      // 1x32 (cloned row) OR empty if unknown
    int       hostKF = -1;
    int       seen   = 0;
    int       found  = 0;
    float     invScale = 1.f; // quick gating by distance (optional)
    bool alive = true;
  };

  bool mapInitialized_ = false;
  std::vector<Keyframe>  kfs_;
  std::vector<MapPoint>  mps_;
  int nextKFId_ = 0;

  // Working buffers reused each frame (avoid allocs in hot loop)
  std::vector<int>     pnp_indices_;     // indices into mps_
  std::vector<cv::Point2f> pnp_pixels_;  // matched 2D
  std::vector<cv::Point3f> pnp_points_;  // 3D (float for cv PnP)
  cv::Mat rvec_, tvec_;                  // current cam pose (cw) for PnP refine

  // KF policy
  double lastKFTs_ = 0.0;
  int    lastKFInliers_ = 0;

  // ====== API ======
  // 2-view init from 2D-2D (processing-scale points)
  bool tryTwoViewInit(const std::vector<cv::Point2f>& prevProcPts,
                      const std::vector<cv::Point2f>& curProcPts);

  // Per-frame 3D-2D tracking using MapPoints (fills Rwc_/twc_ on success)
  bool trackWithPnP();

  // KF insertion + triangulation vs last KF
  bool shouldInsertKF(int pnpInliers, double nowTs) const;
  void insertKeyframeAndTriangulate();

  // Helper: compute ORB at arbitrary pixel locations (processing scale)
  void computeORBAtPoints(const cv::Mat& img,
                          const std::vector<cv::Point2f>& pts,
                          cv::Mat& outDesc);

  // Project MapPoints and collect 3D-2D with small reprojection window
  int harvestPnpCorrespondences(float reprojThreshPx = 8.f, int maxTake = 500);

  // Utility: K (intrinsics) and its inverse at **full-res**
  inline cv::Matx33d K()  const { return cv::Matx33d(fx_, 0,  cx_,
                                                     0,  fy_, cy_,
                                                     0,  0,  1); }
  inline cv::Matx33d Ki() const {
    const double ix = 1.0/std::max(1e-9, fx_);
    const double iy = 1.0/std::max(1e-9, fy_);
    return cv::Matx33d(ix,0,-cx_*ix,  0,iy,-cy_*iy,  0,0,1);
  }

  // Timing
  double t_last_orb_ms_   = 0.0;
  // ===== Visual Odometry (VO) state =====
  // World pose of camera: Rwc_ (3x3), twc_ (3x1); start at identity
  cv::Matx33d Rwc_ = cv::Matx33d::eye();
  cv::Vec3d   twc_ = cv::Vec3d(0,0,0);

  // History of world positions for drawing (x,z used for top-down path)
  std::vector<cv::Point3f> path_; // (x,y,z), push one per frame

  // Integrate relative pose from Essential-matrix inlier correspondences
  void integrateVO_E(const std::vector<cv::Point2f>& prevProcPts,
                     const std::vector<cv::Point2f>& curProcPts);
  // Integrate rotation from Homography (pure rotation/planar scenes)
  void integrateVO_H(const std::vector<cv::Point2f>& prevProcPts,
                     const std::vector<cv::Point2f>& curProcPts);

  // Utility already present:
  // void toFullResPixels(const std::vector<cv::Point2f>& procPts,
  //                      std::vector<cv::Point2f>& fullResPx) const;

  // ===== E/H model gate state (unique names: eh*) =====
  // 0 = NONE, 1 = E (Essential), 2 = H (Homography)
  int    ehModel_        = 0;
  int    ehInliersE_     = 0;
  int    ehInliersH_     = 0;
  double ehParallaxDeg_  = 0.0;
  // --- Rotation prior from last Essential decomposition (used once to seed PnP)
  cv::Matx33d R_delta_prior_ = cv::Matx33d::eye();

  struct ImuSample {
    double ts;
    cv::Vec3d acc;   // m/s^2
    cv::Vec3d gyro;  // rad/s
  };
  
  std::vector<ImuSample> imuBuf_;
  double lastImuFuseTS_ = 0.0;
  cv::Matx33d R_imu_delta_ = cv::Matx33d::eye();   // integrated delta since last fuse
  double t_last_imu_ms_ = 0.0;
  // ---- IMU telemetry for HUD ----
  int    imuSamplesInWindow_ = 0;
  double imuWindowStartTS_   = 0.0;
  double imuHz_              = 0.0;   // computed rate
  int    imuUsedThisFrame_   = 0;     // 0/1 (set in feedFrame)
  int    imuSamplesUsedThisFrame_ = 0;
  double lastImuSampleTS_    = 0.0;
  
  // Optional: gravity direction estimate in world (for later)
  cv::Vec3d g_world_ = cv::Vec3d(0, 9.81, 0);   // m/s^2 (down)

  // Compute E vs H on corresponding point pairs (processing-scale coords)
  void   runEvsHGate(const std::vector<cv::Point2f>& prevProcPts,
                     const std::vector<cv::Point2f>& curProcPts);

  // Utility: build full-res pixel pairs from processing-scale points
  void   toFullResPixels(const std::vector<cv::Point2f>& procPts,
                         std::vector<cv::Point2f>& fullResPx) const;
  
  cv::Matx33d integrateImuDeltaRotationAccCorr(double t0, double t1, bool* used);
};

