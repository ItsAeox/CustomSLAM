#include <emscripten.h>
#include <emscripten/bind.h>
#include "frontend/feature_tracker.h"
#include "map/map.h"
#include "common/camera.h"

using namespace emscripten;

static std::unique_ptr<System> g_sys; // wrap {Camera, Tracker, Map}

void initSystem(int width, int height, double fx, double fy, double cx, double cy) {
    g_sys = std::make_unique<System>(Camera(fx, fy, cx, cy, width, height));
}

void feedFrame(uintptr_t imgPtr, double timestamp, int strideBytes, bool isRGBA) {
    g_sys->ProcessFrame(reinterpret_cast<uint8_t*>(imgPtr), timestamp, strideBytes, isRGBA);
}

void feedIMU(double ts, double ax, double ay, double az, double gx, double gy, double gz) {
    g_sys->ProcessIMU(ts, ax, ay, az, gx, gy, gz); // no-op in M0
}

emscripten::val getPose() {
    // returns Float64Array[16] column-major 4x4 (T_wc) for renderer
    std::array<double,16> T = g_sys->CurrentPoseGL();
    return val(typed_memory_view(T.size(), T.data()));
}

EMSCRIPTEN_BINDINGS(vio_module) {
    function("initSystem", &initSystem);
    function("feedFrame", &feedFrame, allow_raw_pointers());
    function("feedIMU", &feedIMU);
    function("getPose", &getPose);
    }
