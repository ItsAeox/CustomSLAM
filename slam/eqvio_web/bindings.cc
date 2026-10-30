// JavaScript API of the EqVIO mode (web/eqvio.worker.js), conventions as slam/port/web/bindings.cc:
//   const vio = new Module.WebEqvio(configYaml, [fx, fy, cx, cy], w, h, Tbc);   // Tbc: 16 numbers, row-major
//   vio.addImu(t, gx, gy, gz, ax, ay, az);                       // IMU clock (s), rad/s and m/s^2 (with gravity)
//   const ptr = vio.frameBuffer(w, h);                          // gray8 buffer inside the wasm heap
//   const ok = vio.track(t);                                    // camera clock (s): the frame in frameBuffer()
//   vio.poseTwc()   -> Float32Array(16), column-major camera-to-world (gravity-aligned world, metres)
//   vio.points(keepSeconds) -> Float32Array(3N): the filter's landmarks in the world + those dropped within keepSeconds
//   vio.stats()     -> Float64Array [initialised, features, GIFT ms, filter ms, speed m/s]
//   vio.reset()     -> a new filter (divergence)
//   Typed arrays are views into the wasm heap: copy them (slice()) before the next call.
#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <chrono>
#include <memory>
#include <vector>

#include "EqvioWeb.h"

using namespace emscripten;

namespace {
std::vector<double> toVector(const val& a) { return vecFromJSArray<double>(a); }
}  // namespace

class WebEqvio {
  public:
    WebEqvio(const std::string& config, const val& intrinsics, int w, int h, const val& Tbc)
        : vio_(config, toVector(intrinsics), w, h, toVector(Tbc)) {}

    void addImu(double t, double gx, double gy, double gz, double ax, double ay, double az) {
        vio_.addImu(t, Eigen::Vector3d(gx, gy, gz), Eigen::Vector3d(ax, ay, az));
    }

    uintptr_t frameBuffer(int w, int h) {
        if (gray_.cols != w || gray_.rows != h) gray_.create(h, w, CV_8UC1);
        return reinterpret_cast<uintptr_t>(gray_.data);
    }

    bool track(double t) {
        vio_.addFrame(t, gray_);
        const Eigen::Matrix4d T = vio_.Twc();
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++) twc_[4 * c + r] = static_cast<float>(T(r, c));
        return vio_.initialised();
    }

    val poseTwc() const { return val(typed_memory_view(16, twc_)); }

    val points(double keepSeconds) {
        pts_.clear();
        for (const auto& p : vio_.points(keepSeconds)) pts_.insert(pts_.end(), {(float)p.x(), (float)p.y(), (float)p.z()});
        return val(typed_memory_view(pts_.size(), pts_.data()));
    }

    val stats() {
        stats_[0] = vio_.initialised();
        stats_[1] = vio_.nFeatures;
        stats_[2] = vio_.msFeatures;
        stats_[3] = vio_.msFilter;
        stats_[4] = vio_.speed();
        return val(typed_memory_view(5, stats_));
    }

    void reset() { vio_.reset(); }
    double cameraLag() const { return vio_.cameraLag; }

  private:
    EqvioWeb vio_;
    cv::Mat gray_;
    float twc_[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::vector<float> pts_;
    double stats_[5] = {0, 0, 0, 0, 0};
};

EMSCRIPTEN_BINDINGS(eqvio_web) {
    class_<WebEqvio>("WebEqvio")
        .constructor<const std::string&, const val&, int, int, const val&>()
        .function("addImu", &WebEqvio::addImu)
        .function("frameBuffer", &WebEqvio::frameBuffer)
        .function("track", &WebEqvio::track)
        .function("poseTwc", &WebEqvio::poseTwc)
        .function("points", &WebEqvio::points)
        .function("stats", &WebEqvio::stats)
        .function("reset", &WebEqvio::reset)
        .function("cameraLag", &WebEqvio::cameraLag);
}
