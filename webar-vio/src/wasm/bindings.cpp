#include <memory>
#include <vector>
#include <emscripten/bind.h>
#include "system.h"
#include "common/camera.h"
using namespace emscripten;

static std::unique_ptr<System> g_sys;

void initSystem(int width, int height, double fx, double fy, double cx, double cy) {
  g_sys = std::make_unique<System>(Camera(fx, fy, cx, cy, width, height));
}

void feedFrame(uintptr_t imgPtr, double ts, int strideBytes, bool isRGBA) {
  if (!g_sys) return;
  g_sys->ProcessFrame(reinterpret_cast<uint8_t*>(imgPtr), ts, strideBytes, isRGBA);
}

// JS-friendly: Uint8Array -> std::vector<uint8_t>
void feedFrameJS(val u8array, double ts, int width, int height, bool isRGBA) {
  if (!g_sys) return;
  const int strideBytes = width * (isRGBA ? 4 : 1);
  std::vector<uint8_t> buf = convertJSArrayToNumberVector<uint8_t>(u8array);
  g_sys->ProcessFrame(buf.data(), ts, strideBytes, isRGBA);
}

void feedIMU(double ts, double ax, double ay, double az, double gx, double gy, double gz) {
  if (!g_sys) return;
  g_sys->ProcessIMU(ts, ax, ay, az, gx, gy, gz);
}

// Return a copy (safe lifetime for JS)
// std::vector<double> getPoseVec() {
//   if (!g_sys) return {};
//   auto T = g_sys->CurrentPoseGL(); // std::array<double,16> or similar
//   return std::vector<double>(T.begin(), T.end());
// }

std::vector<double> getPoseVec() {
  if (!g_sys) return {};
  auto T = g_sys->CurrentPoseGLThree();
  return std::vector<double>(T.begin(), T.end());
}

int getNumKeypoints() { return g_sys ? g_sys->num_keypoints_  : -1; }
int getNumInliers()   { return g_sys ? g_sys->num_inliers_    : -1; }
int getTrackState()   { return g_sys ? g_sys->tracking_state_ : 0; }

EMSCRIPTEN_BINDINGS(vio_module) {
  function("initSystem", &initSystem);
  function("feedFrame", &feedFrame, allow_raw_pointers());
  function("feedFrameJS", &feedFrameJS);
  function("feedIMU", &feedIMU);
  function("getNumKeypoints", &getNumKeypoints);
  function("getNumInliers",   &getNumInliers);
  function("getTrackState",   &getTrackState);

  // Register vector<double> so Embind can marshal it
  register_vector<double>("VectorDouble");

  function("getPose", &getPoseVec);
}
