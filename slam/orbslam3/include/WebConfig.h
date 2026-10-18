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
    int lbaFast = 0;           // Web.LBA.Fast: local BA of monocular visual-only maps without a g2o graph (Optimizer.cc
                               // WebBASolveLM): 1 = used, 2 = both solved, g2o kept, differences reported (check)
    int mappingNeighbors = 0;  // Web.Mapping.Neighbors: covisible keyframes for triangulation / fusion (0 = stock: 30 mono)
    int descriptor = 0;        // Web.Descriptor: 0 = ORB (rotated BRIEF, stock), 1 = TEBLID-256 (needs a TEBLID vocabulary)
    int orbParallel = 0;       // Web.ORB.Parallel: the pyramid levels of ORB extraction (FAST + octree + orientation,
                               // blur + descriptors) run on this many threads (1 = 4 threads; 0 = serial, stock); same
                               // keypoints and descriptors. Own pool: OpenCV's WASM build has no parallel framework
    int orbSSC = 0;            // Web.ORB.SSC: 1 = keypoints per level chosen by Suppression via Square Covering (Bailo et
                               // al. 2018: strongest corners, homogeneous spread) instead of the octree (stock)
    double relocTimeout = 3.0; // Web.Track.RelocTimeout: seconds of relocalisation in the current map before a new map (stock 3)
    int rotFar = 0;            // Web.Rot.FarPoints: 1 = pure-rotation prototype (LocalMapping::WebCreateFarPoints): keyframe
                               // pairs too close to triangulate give far points (on the ray at FarDepth x the scene median
                               // depth), kept out of local BA, fusion and the AR plane fits. 2 = the same, but in the
                               // pose solves (g2o + fast) a far point is a point at infinity: its mean viewing direction,
                               // moved by the rotation only (mode 1 kept the made-up depth: rot_screen ATE x1.66). 3 = mode 2
                               // and far points also kept out of relocalisation PnP, Sim3 (loop / merge) and keyframe-culling
                               // counts (mode 2 still broke room1 / MH04 there). 0 = stock
    double rotFarDepth = 10.0; // Web.Rot.FarDepth: far-point depth in scene median depths
    int rotFarLife = 6;        // Web.Rot.FarLife: keyframes a far point lives (it constrains rotation only while the camera
                               // stays near where it was made)
    double kfCoverage = 0;     // Web.KF.Coverage: > 0 = also insert a keyframe (even while local mapping is busy, queue < 2)
                               // when this share of the textured image cells (4x4 grid, >= 3 keypoints) holds no tracked map
                               // point: new textured area coming into view (turning toward unmapped parts). 0 = stock
    double triMinBaseline = 0.01;  // Web.Map.MinBaseline: monocular triangulation needs baseline / scene depth >= this (stock 0.01)
    double triMaxCosParallax = 0.9998;  // Web.Map.MaxCosParallax: and the rays' cos(parallax) below this (stock 0.9998, ~1.1 deg)
    double mapLoadWait = 20.0; // Web.Map.LoadWait: after a room map is loaded (System::WebLoadMap), seconds of relocalisation
                               // into it before a new map is started; < 0 = wait until relocalised
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
    int kltParallel = 0;       // Web.KLT.Parallel: 1 = the LK points are split over the shared thread pool (WebPool.h; same
                               // result: every point is tracked on its own); 2 = both, serial result kept, mismatches and
                               // timings reported (check). OpenCV's WASM build runs LK on one thread
    int kltNormalize = 0;      // Web.KLT.Normalize: 1 = the LK pyramid is built from the image with its global mean and
                               // contrast normalised (128 / 48): cancels exposure (gain/offset) changes between frames,
                               // which plain LK's brightness-constancy model does not handle. ORB frames are unaffected

    int bowLevelsUp = 4;       // Web.BoW.LevelsUp: tree levels above the leaves for the BoW direct index (FeatureVector)
                               // used by SearchByBoW (Frame/KeyFrame::ComputeBoW; stock 4, made for L = 6 trees). A smaller
                               // vocabulary needs L - 2 to index at the same depth (100 nodes for k = 10).

    // Keyframe policy (monocular, visual-only). 0 = stock. Fewer keyframes -> less mapping work and memory (G12, G14).
    int kfBusyQueue = 0;       // Web.KF.BusyQueue: monocular, N > 0 = a keyframe that is due is queued while local mapping is
                               // busy (up to N waiting), as stock does for stereo; stock mono refuses it, so mapping that
                               // cannot keep up (the phone) stops the map from growing (ISSUES G22). Queued keyframes get
                               // their points triangulated; local BA, neighbour fusion and culling wait until mapping
                               // catches up (LocalMapping skips them while keyframes are queued)
    int mapFuseQueued = 0;     // Web.Mapping.FuseQueued: 1 = neighbour search / point fusion also while keyframes are queued
                               // (stock skips it then, with local BA); only local BA waits for mapping to catch up
    double kfRefRatio = 0;     // Web.KF.RefRatio: insert a keyframe when tracked map points < ratio x the reference
                               // keyframe's (Tracking::NeedNewKeyFrame; stock mono 0.9: lower = fewer keyframes)
    double cullRedundancy = 0; // Web.Mapping.CullRedundancy: cull a keyframe whose points are seen by >= 3 other
                               // keyframes for more than this fraction (LocalMapping::KeyFrameCulling; stock 0.9: lower
                               // = more culling)

    // Gyro-aided prediction (visual-only modes; see Tracking::WebMotionPrediction). Off by default = stock.
    int gyroPrior = 0;         // Web.Gyro.Prior: 1 = the rotation of the frame-to-frame prediction (motion model and
                               // KLT initial flow) comes from the gyroscope integrated between the two frames
    double gyroTimeOffset = 0; // Web.Gyro.TimeOffset: seconds added to frame times to get IMU times (> 0: IMU stamps late)
    float gyroRbc[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // camera -> IMU body rotation, row-major (from IMU.T_b_c1)

    // Loss recovery by coasting (monocular, visual-only; see Tracking::WebCoastTrack). Off by default = stock.
    int coast = 0;             // Web.Track.Coast: 1 = while recently lost, search the local map around the last good pose
                               // (rotated by the gyro when Web.Gyro.Prior is on) before BoW relocalisation; 2 = also, if
                               // that fails, match the frame against the whole local map by descriptor + MLPnP RANSAC
    double coastMaxTime = 1.5; // Web.Track.CoastTime: seconds after the last good frame during which coasting is tried
    int coastMinInliers = 50;  // Web.Track.CoastMinInliers: pose inliers to accept the coasted pose (relocalisation: 50)

    // Turn bridge (monocular, visual-only, needs Web.Gyro.Prior; see Tracking::WebTurnBridge). Off by default = stock.
    // While lost after turning away (gyro rotation from the last good pose >= MinAngle, coasting and relocalisation
    // failing), a two-view reconstruction of the new view is added to the CURRENT map at the gyro-propagated pose
    // (camera centre held), scaled to the last reference keyframe's median depth, hung off that keyframe in the
    // spanning tree: tracking resumes in the same map instead of waiting for the relocalisation timeout.
    int turnBridge = 0;          // Web.Turn.Bridge: 1 = stock two-view reconstruction, accepted only if its rotation agrees
                                 // with the gyro's (MaxRotDiff); 2 = the gyro two-view solve (WebGyroTwoView, Web.Init.Gyro)
    double turnMinAngle = 20;    // Web.Turn.MinAngle: degrees turned away from the last good pose before bridging
    double turnMaxRotDiff = 3;   // Web.Turn.MaxRotDiff: mode 1, degrees between the reconstruction's and the gyro's rotation

    int fastPose = 0;          // Web.Track.FastPose: motion-only pose optimisation of monocular frames (Optimizer.cc)
                               // 0 = g2o (stock), 1 = the same problem solved without a g2o graph (WebPoseOptimization),
                               // 2 = both on every call, g2o's result kept, differences and timings reported (check)
    int fastPoseWarm = 0;      // Web.Track.FastPoseWarm: 1 = rounds 2-4 start from the previous round's pose (stock and
                               // FastPose 1 restart every round from the initial pose)

    // Monocular initialisation (two-view map), stock values by default. Lower = earlier first map, maybe a worse one.
    int initMinMatches = 100;     // Web.Init.MinMatches: matches between the reference and current frame (else restart)
    double initMinParallax = 1.0; // Web.Init.MinParallax: degrees (TwoViewReconstruction)
    int initMinTriangulated = 50; // Web.Init.MinTriangulated: triangulated points of the two-view solution
    int initGyro = 0;             // Web.Init.Gyro: 1 = two-view initialisation with the gyro rotation (2-point RANSAC on
                                  // the translation direction; needs Web.Gyro.Prior), stock reconstruction as fallback;
                                  // 2 = also refine rotation + translation (maps later in ginit_ab: not used)
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
    readInt("Web.LBA.Fast", c.lbaFast);
    readInt("Web.Mapping.Neighbors", c.mappingNeighbors);
    readInt("Web.Descriptor", c.descriptor);
    readInt("Web.ORB.Parallel", c.orbParallel);
    readInt("Web.ORB.SSC", c.orbSSC);
    readInt("Web.Track.KeepMap", c.keepMap);
    readInt("Web.Track.TimeScaledMotion", c.timeScaledMotion);
    readInt("Web.KLT.Enable", c.kltEnable);
    readInt("Web.KLT.MaxConsecutive", c.kltMaxConsecutive);
    readInt("Web.KLT.MinTracked", c.kltMinTracked);
    readInt("Web.KLT.MinInliers", c.kltMinInliers);
    readInt("Web.KLT.WinSize", c.kltWinSize);
    readInt("Web.KLT.Levels", c.kltLevels);
    readInt("Web.KLT.Iterations", c.kltIterations);
    readInt("Web.KLT.Normalize", c.kltNormalize);
    readInt("Web.KLT.Parallel", c.kltParallel);
    cv::FileNode rt = fs["Web.Track.RelocTimeout"];
    if (!rt.empty() && (rt.isInt() || rt.isReal())) c.relocTimeout = rt.real();
    readInt("Web.Rot.FarPoints", c.rotFar);
    readInt("Web.Rot.FarLife", c.rotFarLife);
    cv::FileNode fd = fs["Web.Rot.FarDepth"];
    if (!fd.empty() && (fd.isInt() || fd.isReal())) c.rotFarDepth = fd.real();
    for (auto [key, v] : {std::pair<const char*, double*>{"Web.KF.Coverage", &c.kfCoverage}, {"Web.Map.MinBaseline", &c.triMinBaseline},
                           {"Web.Map.MaxCosParallax", &c.triMaxCosParallax}}) {
        cv::FileNode n = fs[key];
        if (!n.empty() && (n.isInt() || n.isReal())) *v = n.real();
    }
    cv::FileNode lw = fs["Web.Map.LoadWait"];
    if (!lw.empty() && (lw.isInt() || lw.isReal())) c.mapLoadWait = lw.real();
    cv::FileNode mf = fs["Web.KLT.MaxFlow"];
    if (!mf.empty() && (mf.isInt() || mf.isReal())) c.kltMaxFlow = mf.real();
    cv::FileNode mi = fs["Web.KLT.MinInlierFrac"];
    if (!mi.empty() && (mi.isInt() || mi.isReal())) c.kltMinInlierFrac = mi.real();
    cv::FileNode fb = fs["Web.KLT.FBThreshold"];
    if (!fb.empty() && (fb.isInt() || fb.isReal())) c.kltFBThreshold = fb.real();
    readInt("Web.BoW.LevelsUp", c.bowLevelsUp);
    cv::FileNode kr = fs["Web.KF.RefRatio"];
    if (!kr.empty() && (kr.isInt() || kr.isReal())) c.kfRefRatio = kr.real();
    cv::FileNode cr = fs["Web.Mapping.CullRedundancy"];
    if (!cr.empty() && (cr.isInt() || cr.isReal())) c.cullRedundancy = cr.real();
    readInt("Web.KF.BusyQueue", c.kfBusyQueue);
    readInt("Web.Mapping.FuseQueued", c.mapFuseQueued);
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
    readInt("Web.Track.Coast", c.coast);
    cv::FileNode ct = fs["Web.Track.CoastTime"];
    if (!ct.empty() && (ct.isInt() || ct.isReal())) c.coastMaxTime = ct.real();
    readInt("Web.Track.CoastMinInliers", c.coastMinInliers);
    readInt("Web.Turn.Bridge", c.turnBridge);
    for (auto [key, v] : {std::pair<const char*, double*>{"Web.Turn.MinAngle", &c.turnMinAngle}, {"Web.Turn.MaxRotDiff", &c.turnMaxRotDiff}}) {
        cv::FileNode n = fs[key];
        if (!n.empty() && (n.isInt() || n.isReal())) *v = n.real();
    }
    if (c.turnBridge && !c.gyroPrior) {
        std::cout << "Web config: Web.Turn.Bridge needs Web.Gyro.Prior; turn bridge off" << std::endl;
        c.turnBridge = 0;
    }
    readInt("Web.Track.FastPose", c.fastPose);
    readInt("Web.Track.FastPoseWarm", c.fastPoseWarm);
    readInt("Web.Init.MinMatches", c.initMinMatches);
    cv::FileNode ip = fs["Web.Init.MinParallax"];
    if (!ip.empty() && (ip.isInt() || ip.isReal())) c.initMinParallax = ip.real();
    readInt("Web.Init.MinTriangulated", c.initMinTriangulated);
    readInt("Web.Init.Gyro", c.initGyro);
    std::cout << "Web config: LBA.MaxOptKFs " << c.lbaMaxOptKFs << ", LBA.MaxFixedKFs " << c.lbaMaxFixedKFs
              << ", LBA.Iterations " << c.lbaIterations << ", Mapping.Neighbors " << c.mappingNeighbors
              << " (0 = stock); descriptor " << (c.descriptor == 1 ? "TEBLID-256" : "ORB") << "; reloc timeout "
              << c.relocTimeout << " s" << (c.keepMap ? " (keep map)" : "") << (c.timeScaledMotion ? "; time-scaled motion" : "")
              << (c.gyroPrior ? "; gyro prior (IMU time offset " + std::to_string(c.gyroTimeOffset * 1000) + " ms)" : std::string())
              << (c.bowLevelsUp != 4 ? "; BoW levels up " + std::to_string(c.bowLevelsUp) : std::string())
              << (c.kfRefRatio > 0 ? "; KF ref ratio " + std::to_string(c.kfRefRatio) : std::string())
              << (c.kfBusyQueue > 0 ? "; keyframes queued while mapping is busy (up to " + std::to_string(c.kfBusyQueue) + ")" : std::string())
              << (c.mapFuseQueued ? "; fusion also for queued keyframes" : "")
              << (c.lbaFast ? (c.lbaFast == 1 ? "; fast local BA" : "; fast local BA (compare with g2o)") : "")
              << (c.cullRedundancy > 0 ? "; KF cull redundancy " + std::to_string(c.cullRedundancy) : std::string())
              << (c.coast ? "; coast " + std::to_string(c.coastMaxTime) + " s, " + std::to_string(c.coastMinInliers) + " inliers"
                          : std::string())
              << (c.fastPose ? (c.fastPose == 1 ? "; fast pose solver" : "; fast pose solver (compare with g2o)") : "")
              << (c.fastPose && c.fastPoseWarm ? " (warm start)" : "")
              << (c.orbSSC ? "; SSC keypoint selection" : "")
              << (c.orbParallel ? "; parallel ORB levels (" + std::to_string(c.orbParallel == 1 ? 4 : c.orbParallel) + " threads)" : "")
              << (c.initMinMatches != 100 || c.initMinParallax != 1.0 || c.initMinTriangulated != 50
                      ? "; init " + std::to_string(c.initMinMatches) + " matches, " + std::to_string(c.initMinParallax) +
                            " deg, " + std::to_string(c.initMinTriangulated) + " points"
                      : std::string())
              << (c.initGyro ? "; gyro two-view init" : "")
              << (c.turnBridge ? std::string("; turn bridge (") + (c.turnBridge == 1 ? "two-view + gyro check " + std::to_string(c.turnMaxRotDiff) + " deg"
                                                                                     : std::string("gyro two-view")) +
                                     ", after " + std::to_string(c.turnMinAngle) + " deg)"
                               : std::string())
              << "; KLT " << (c.kltEnable ? "on" : "off");
    if (c.kltEnable)
        std::cout << " (max consecutive " << c.kltMaxConsecutive << ", min tracked " << c.kltMinTracked
                  << ", min inliers " << c.kltMinInliers << ", win " << c.kltWinSize << ", levels " << c.kltLevels
                  << ", FB " << c.kltFBThreshold << " px, iterations " << c.kltIterations << ", max flow " << c.kltMaxFlow
                  << " px, min inlier frac " << c.kltMinInlierFrac << (c.kltNormalize ? ", exposure-normalised" : "")
                  << (c.kltParallel ? (c.kltParallel == 1 ? ", LK on 4 threads" : ", LK serial vs 4 threads (compare)") : "") << ")";
    std::cout << std::endl;
}

}  // namespace ORB_SLAM3

#endif
