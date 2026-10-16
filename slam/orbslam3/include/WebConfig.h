// ORBSLAM3_WEB (web port, not upstream): runtime knobs for the browser build, read from optional "Web.*" keys in the
// settings file when the System starts. Every default reproduces stock ORB-SLAM3 exactly, so a settings file without
// Web.* keys runs the unmodified algorithm (used as the baseline in every comparison).
#ifndef ORBSLAM3_WEBCONFIG_H
#define ORBSLAM3_WEBCONFIG_H

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

namespace ORB_SLAM3
{

struct WebConfig
{
    int lbaMaxOptKFs = 0;      // Web.LBA.MaxOptKFs: optimised keyframes in local BA, best-covisible first (0 = all, stock)
    int lbaMaxFixedKFs = 0;    // Web.LBA.MaxFixedKFs: fixed keyframes, most observations of local points first (0 = all, stock)
    int lbaIterations = 10;    // Web.LBA.Iterations: Levenberg-Marquardt iterations of local BA (stock 10)
    int mappingNeighbors = 0;  // Web.Mapping.Neighbors: covisible keyframes for triangulation / fusion (0 = stock: 30 mono)
    int descriptor = 0;        // Web.Descriptor: 0 = ORB (rotated BRIEF, stock), 1 = TEBLID-256 (needs a TEBLID vocabulary)
    double relocTimeout = 3.0; // Web.Track.RelocTimeout: seconds of relocalisation in the current map before a new map (stock 3)
    int timeScaledMotion = 0;  // Web.Track.TimeScaledMotion: 1 = the constant-velocity prediction is rescaled to the
                               // actual frame interval (live cameras drop frames when tracking is slow). 0 = stock
    int keepMap = 0;           // Web.Track.KeepMap: 1 = a tracking loss never resets or replaces the active map; the tracker
                               // relocalises in it until the camera returns (AR anchors stay valid). Also set at runtime
                               // by the AR page while a cactus is placed (bindings: setKeepMap). 0 = stock.

    // KLT tracking frames (monocular; see Tracking::TrackKLT). Off by default = stock ORB-SLAM3 (ORB on every frame).
    int kltEnable = 0;         // Web.KLT.Enable: 1 = track some frames with pyramidal LK instead of ORB extraction
    int kltMaxConsecutive = 2; // Web.KLT.MaxConsecutive: at most this many KLT frames between two ORB frames
    int kltMinTracked = 60;    // Web.KLT.MinTracked: map-point tracks that must survive LK + forward-backward check
    int kltMinInliers = 40;    // Web.KLT.MinInliers: pose-optimisation inliers to accept the KLT pose
    int kltWinSize = 15;       // Web.KLT.WinSize: LK window (pixels). 15 vs 21: LK 2.2x cheaper in WASM, same ATE (MH01)
    int kltLevels = 2;         // Web.KLT.Levels: LK pyramid levels above full resolution (flow is predicted, so 2 suffice)
    double kltFBThreshold = 1.0;  // Web.KLT.FBThreshold: max forward-backward error (pixels); <= 0: no backward pass
    int kltIterations = 30;    // Web.KLT.Iterations: max LK iterations per pyramid level
    double kltMaxFlow = 0;     // Web.KLT.MaxFlow: use ORB when the predicted median image motion exceeds this (px; 0 = off)
    double kltMinInlierFrac = 0;  // Web.KLT.MinInlierFrac: KLT pose inliers must reach this fraction of the last ORB
                               // frame's inliers, else the frame is re-tracked with ORB (0 = off)

    // Gyro-aided prediction (visual-only modes; see Tracking::WebMotionPrediction). Off by default = stock.
    int gyroPrior = 0;         // Web.Gyro.Prior: 1 = the rotation of the frame-to-frame prediction (motion model and
                               // KLT initial flow) comes from the gyroscope integrated between the two frames
    double gyroTimeOffset = 0; // Web.Gyro.TimeOffset: seconds added to frame times to get IMU times (> 0: IMU stamps late)
    float gyroRbc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // camera -> IMU body rotation, row-major (from IMU.T_b_c1)
};

inline WebConfig& webConfig()
{
    static WebConfig cfg;
    return cfg;
}

inline void loadWebConfig(const std::string& settingsFile)
{
    WebConfig& c = webConfig();
    c = WebConfig();
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if (!fs.isOpened()) return;
    auto readInt = [&fs](const char* key, int& v) {
        cv::FileNode n = fs[key];
        if (!n.empty() && (n.isInt() || n.isReal())) v = (int)n.real();
    };
    readInt("Web.LBA.MaxOptKFs", c.lbaMaxOptKFs);
    readInt("Web.LBA.MaxFixedKFs", c.lbaMaxFixedKFs);
    readInt("Web.LBA.Iterations", c.lbaIterations);
    readInt("Web.Mapping.Neighbors", c.mappingNeighbors);
    readInt("Web.Descriptor", c.descriptor);
    readInt("Web.Track.KeepMap", c.keepMap);
    readInt("Web.Track.TimeScaledMotion", c.timeScaledMotion);
    readInt("Web.KLT.Enable", c.kltEnable);
    readInt("Web.KLT.MaxConsecutive", c.kltMaxConsecutive);
    readInt("Web.KLT.MinTracked", c.kltMinTracked);
    readInt("Web.KLT.MinInliers", c.kltMinInliers);
    readInt("Web.KLT.WinSize", c.kltWinSize);
    readInt("Web.KLT.Levels", c.kltLevels);
    readInt("Web.KLT.Iterations", c.kltIterations);
    cv::FileNode rt = fs["Web.Track.RelocTimeout"];
    if (!rt.empty() && (rt.isInt() || rt.isReal())) c.relocTimeout = rt.real();
    cv::FileNode mf = fs["Web.KLT.MaxFlow"];
    if (!mf.empty() && (mf.isInt() || mf.isReal())) c.kltMaxFlow = mf.real();
    cv::FileNode mi = fs["Web.KLT.MinInlierFrac"];
    if (!mi.empty() && (mi.isInt() || mi.isReal())) c.kltMinInlierFrac = mi.real();
    cv::FileNode fb = fs["Web.KLT.FBThreshold"];
    if (!fb.empty() && (fb.isInt() || fb.isReal())) c.kltFBThreshold = fb.real();
    readInt("Web.Gyro.Prior", c.gyroPrior);
    cv::FileNode go = fs["Web.Gyro.TimeOffset"];
    if (!go.empty() && (go.isInt() || go.isReal())) c.gyroTimeOffset = go.real();
    if (c.gyroPrior) {  // camera -> body rotation from the IMU extrinsics (same key as ORB-SLAM3's inertial settings)
        cv::Mat Tbc;
        fs["IMU.T_b_c1"] >> Tbc;
        if (Tbc.rows >= 3 && Tbc.cols >= 3) {
            Tbc.convertTo(Tbc, CV_32F);
            for (int r = 0; r < 3; r++)
                for (int k = 0; k < 3; k++) c.gyroRbc[3 * r + k] = Tbc.at<float>(r, k);
        } else {
            std::cout << "Web config: Web.Gyro.Prior needs IMU.T_b_c1 (camera -> IMU extrinsics); gyro prior off" << std::endl;
            c.gyroPrior = 0;
        }
    }
    std::cout << "Web config: LBA.MaxOptKFs " << c.lbaMaxOptKFs << ", LBA.MaxFixedKFs " << c.lbaMaxFixedKFs
              << ", LBA.Iterations " << c.lbaIterations << ", Mapping.Neighbors " << c.mappingNeighbors
              << " (0 = stock); descriptor " << (c.descriptor == 1 ? "TEBLID-256" : "ORB") << "; reloc timeout "
              << c.relocTimeout << " s" << (c.keepMap ? " (keep map)" : "") << (c.timeScaledMotion ? "; time-scaled motion" : "")
              << (c.gyroPrior ? "; gyro prior (IMU time offset " + std::to_string(c.gyroTimeOffset * 1000) + " ms)" : std::string())
              << "; KLT " << (c.kltEnable ? "on" : "off");
    if (c.kltEnable)
        std::cout << " (max consecutive " << c.kltMaxConsecutive << ", min tracked " << c.kltMinTracked
                  << ", min inliers " << c.kltMinInliers << ", win " << c.kltWinSize << ", levels " << c.kltLevels
                  << ", FB " << c.kltFBThreshold << " px, iterations " << c.kltIterations << ", max flow " << c.kltMaxFlow
                  << " px, min inlier frac " << c.kltMinInlierFrac << ")";
    std::cout << std::endl;
}

}  // namespace ORB_SLAM3

#endif
