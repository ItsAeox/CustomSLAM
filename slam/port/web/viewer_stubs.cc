// Web build replacements for ORB-SLAM3's MapDrawer.cc and Viewer.cc (Pangolin/OpenGL viewer).
// Same class interfaces as upstream (include/MapDrawer.h, include/Viewer.h); drawing is a no-op.
// MapDrawer still records the current camera pose because Tracking reports it there.

#include "MapDrawer.h"
#include "Viewer.h"

namespace ORB_SLAM3
{

// ---------------------------------------------------------------- MapDrawer

MapDrawer::MapDrawer(Atlas* pAtlas, const string& /*strSettingPath*/, Settings* /*settings*/) : mpAtlas(pAtlas)
{
    mKeyFrameSize = mKeyFrameLineWidth = mGraphLineWidth = mPointSize = mCameraSize = mCameraLineWidth = 0.f;
}

void MapDrawer::newParameterLoader(Settings* /*settings*/) {}
bool MapDrawer::ParseViewerParamFile(cv::FileStorage& /*fSettings*/) { return true; }
void MapDrawer::DrawMapPoints() {}
void MapDrawer::DrawKeyFrames(const bool, const bool, const bool, const bool) {}
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix& /*Twc*/) {}
void MapDrawer::SetReferenceKeyFrame(KeyFrame* /*pKF*/) {}

void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f& Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix& M, pangolin::OpenGlMatrix& MOw)
{
    M.SetIdentity();
    MOw.SetIdentity();
}

// ---------------------------------------------------------------- Viewer (never constructed when bUseViewer=false)

Viewer::Viewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking* pTracking,
               const string& /*strSettingPath*/, Settings* /*settings*/)
    : both(false), mpSystem(pSystem), mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpTracker(pTracking),
      mT(0), mImageWidth(0), mImageHeight(0), mImageViewerScale(1), mViewpointX(0), mViewpointY(0), mViewpointZ(0),
      mViewpointF(0), mbFinishRequested(false), mbFinished(true), mbStopped(true), mbStopRequested(false),
      mbStopTrack(false)
{
}

void Viewer::newParameterLoader(Settings* /*settings*/) {}
bool Viewer::ParseViewerParamFile(cv::FileStorage& /*fSettings*/) { return true; }
void Viewer::Run() { SetFinish(); }
void Viewer::RequestFinish() { unique_lock<mutex> lock(mMutexFinish); mbFinishRequested = true; }
bool Viewer::CheckFinish() { unique_lock<mutex> lock(mMutexFinish); return mbFinishRequested; }
void Viewer::SetFinish() { unique_lock<mutex> lock(mMutexFinish); mbFinished = true; }
bool Viewer::isFinished() { unique_lock<mutex> lock(mMutexFinish); return mbFinished; }
void Viewer::RequestStop() { unique_lock<mutex> lock(mMutexStop); mbStopRequested = true; }
bool Viewer::isStopped() { return true; }
bool Viewer::isStepByStep() { return false; }
bool Viewer::Stop() { return true; }
void Viewer::Release() { unique_lock<mutex> lock(mMutexStop); mbStopped = false; mbStopRequested = false; }

}  // namespace ORB_SLAM3
