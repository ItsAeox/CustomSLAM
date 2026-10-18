/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Settings.h"

#include "GeometricCamera.h"

#include <mutex>
#include <unordered_set>

namespace ORB_SLAM3
{

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetStepByStep(bool bSet);
    bool GetStepByStep();

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetNumberDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;
#ifdef ORBSLAM3_WEB
    // web port: KLT tracking frames (Web.KLT.*, see TrackKLT). Public for the browser bindings and statistics.
    bool mbWebKLTFrame = false;        // the current frame was tracked by optical flow (no ORB extraction)
    long mnWebKLTFrames = 0, mnWebORBFrames = 0, mnWebKLTFallbacks = 0;
    double mdWebKLTms = 0;             // total time spent in accepted KLT frames (ms)
    double mdWebKLTPyrMs = 0;          // LK pyramid of every image while KLT is enabled (ms, all frames)
    double mdWebKLTFlowMs = 0, mdWebKLTCopyMs = 0, mdWebKLTPoseMs = 0;  // accepted KLT frames: LK, frame build, pose
    long mnWebCoastTries = 0, mnWebCoastRecoveries = 0, mnWebRelocRecoveries = 0;  // Web.Track.Coast statistics
    long mnWebLKCalls = 0, mnWebLKPoints = 0, mnWebLKMismatch = 0;  // Web.KLT.Parallel compare mode
    long mnWebCovKFs = 0;  // keyframes requested by Web.KF.Coverage
    long mnWebKFBusy = 0;  // NeedNewKeyFrame: a keyframe was due while local mapping was busy (diagnostics, --trace)
    size_t WebLocalMapSize() const { return mvpLocalMapPoints.size(); }  // diagnostics (orbslam3_run --trace)
    double mdWebLKSerialMs = 0, mdWebLKParallelMs = 0;
    // room map reload (System::WebLoadMap): the first frames relocalise into the loaded map instead of initialising
    void WebAdoptLoadedMap(Map* pMap);
    bool mbWebLoadedMap = false;       // a map was loaded at the start
    bool mbWebLoadWait = false;        // ... and not relocalised into yet
    KeyFrame* mpWebLoadRefKF = nullptr;
    double mdWebLoadStart = -1;        // first frame time after the load (s)
    double mdWebLoadRelocTime = -1;    // time to the first relocalisation into the loaded map (s; -1 = none)
#endif

    cv::Mat mImGray;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
#ifdef ORBSLAM3_WEB
    // web port: track the current image by pyramidal LK from the last frame's map-point observations + motion-only BA.
    // Returns true and sets mCurrentFrame (a descriptor-free frame with pose) on success; false -> use ORB.
    bool TrackKLT(const cv::Mat& im, double timestamp);
    std::vector<cv::Mat> mvWebPrevPyr, mvWebCurPyr;  // LK pyramids of the previous / current image
    bool mbWebForceORB = false;        // a keyframe was needed on a KLT frame: the next frame must be an ORB frame
    int mnWebConsecutiveKLT = 0;
    int mnWebLastORBInliers = 0;      // map-point inliers of the last ORB-tracked frame (quality gate)
#endif
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;
#ifdef ORBSLAM3_WEB
    double mWebVelocityDt = 0;  // interval (s) of the two frames that define mVelocity
    // mVelocity, rescaled to tNow (Web.Track.TimeScaledMotion), its rotation from the gyro (Web.Gyro.Prior)
    Sophus::SE3f WebMotionPrediction(double tNow) const;
    std::vector<IMU::Point> mvWebGyro;  // gyro samples for Web.Gyro.Prior (IMU clock, oldest first)
    bool WebGyroRotation(double t0, double t1, Eigen::Matrix3f& dR) const;  // body rotation t0 -> t1 from mvWebGyro
    // last frame tracked OK (pose, time, map), the origin of Web.Track.Coast
    bool mbWebGood = false;
    Sophus::SE3f mWebGoodTcw;
    double mdWebGoodTime = 0;
    Map* mpWebGoodMap = nullptr;
    bool mbWebGoodVel = false;     // the motion model at that frame (velocity and its frame interval)
    Sophus::SE3f mWebGoodVel;
    double mdWebGoodVelDt = 0;
    bool WebCoastTrack();
    bool WebGyroTwoView(Sophus::SE3f& T21, vector<cv::Point3f>& vP3D, vector<bool>& vbTriangulated);  // Web.Init.Gyro
    // Web.Turn.Bridge (WebTurnBridge): state of the current loss episode
    bool WebTurnBridge();
    double mdWebBridgeGood = -1;          // mdWebGoodTime of the episode this state belongs to
    Sophus::SE3f mWebBridgeTcw;           // gyro-propagated pose (rotation from the gyro, camera centre held)
    double mdWebBridgeT = 0;              // its time
    bool mbWebBridgeOk = false;           // gyro coverage so far, same map
    bool mbWebBridgeRef = false;          // a reference frame is set (mInitialFrame, mvbPrevMatched)
    Sophus::SE3f mWebBridgeRefTcw;        // the reference frame's propagated pose
    Sophus::SO3f mWebBridgePrevR;         // propagated rotation at the last matching (search positions pre-rotated from it)
    KeyFrame* mpWebBridgeGoodKF = nullptr;  // reference keyframe at the loss: parent of the bridge in the spanning tree
    float mfWebBridgeDepth = 0;           // its median scene depth: the scale of the bridge
    unsigned long mnWebBridgeFrame = 0;   // frame of the last bridge: no motion model from the lost frame before it
public:
    long mnWebBridges = 0, mnWebBridgeTries = 0, mnWebBridgeRejects = 0;  // Web.Turn.Bridge statistics
    unsigned long mnWebBridgeAnchorKF = 0;  // reference keyframe id of the latest bridge (0: none yet)
    int WebOldPointsTracked() const;      // tracked points first seen before the latest bridge (-1: no bridge yet)
protected:
public:
    void WebGrabGyro(const std::vector<IMU::Point>& v);  // visual-only modes: IMU samples for the gyro prior
protected:
#endif

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;

    //int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;
    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;

    void newParameterLoader(Settings* settings);

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
