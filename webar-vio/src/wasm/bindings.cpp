#include <emscripten/bind.h>
#include "system.h"

using namespace emscripten;

static System gSys;

static int    GetNumKeypoints() { return gSys.getNumKeypoints(); }
static int    GetTrackState()   { return gSys.getTrackState(); }
static double GetLastTS()       { return gSys.getLastTS(); }
static double GetLastTotalMS() { return gSys.getLastTotalMS(); }
static double GetLastKltMS()   { return gSys.getLastKltMS();   }
static double GetLastSeedMS()  { return gSys.getLastSeedMS();  }
static double GetLastMeanY()           { return gSys.getLastMeanY(); }


// init from JS
void initSystem(int width, int height, double fx, double fy, double cx, double cy) {
  gSys.init(width, height, fx, fy, cx, cy);
}

// feed frame from a Uint8Array
void feedFrameJS(val u8, double ts, int width, int height, bool isRGBA) {
  const size_t expected = (size_t)width * height * (isRGBA ? 4 : 1);
  const size_t n = u8["length"].as<size_t>();
  if (n < expected) return;

  // copy to a vector (simple + safe)
  std::vector<uint8_t> buf(n);
  for (size_t i=0; i<n; ++i) buf[i] = u8[i].as<uint8_t>();

  gSys.feedFrame(buf.data(), ts, width, height, isRGBA);
}

// Zero-copy path: JS writes pixels into Module.HEAPU8 at 'ptr' and calls this.
void feedFramePtr(uintptr_t ptr, double ts, int width, int height, bool isRGBA) {
  // ptr points into WASM heap (HEAPU8). We can read it directly.
  gSys.feedFrame(reinterpret_cast<const uint8_t*>(ptr), ts, width, height, isRGBA);
}

// return JS array [x0,y0, x1,y1, ...]
val getPoints2D() {
  const auto pts = gSys.getPoints2D();
  val jsArr = val::array();
  for (size_t i=0; i<pts.size(); ++i) jsArr.set(i, pts[i]);
  return jsArr;
}

emscripten::val getPoints2D_Typed() {
  static std::vector<float> buf; // lifetime stable
  auto pts = gSys.getPoints2D(); // doubles
  buf.resize(pts.size());
  for (size_t i=0;i<pts.size();++i) buf[i] = static_cast<float>(pts[i]);
  return emscripten::val(emscripten::typed_memory_view(buf.size(), buf.data()));
}

static emscripten::val GetLastProcWH() {
  auto a = gSys.getLastProcWH();
  emscripten::val out = emscripten::val::array();
  out.set(0, a[0]); out.set(1, a[1]);
  return out;
}

EMSCRIPTEN_BINDINGS(vio_bindings_pointtrack) {
  function("initSystem",   &initSystem);
  function("feedFrameJS",  &feedFrameJS);
  function("getPoints2DArray", &getPoints2D);       // old JS array builder
  function("getPoints2D",      &getPoints2D_Typed); // fast typed view
  function("feedFramePtr",      &feedFramePtr);       // NEW zero-copy path
  function("getLastTotalMS",    &GetLastTotalMS);     // NEW timers
  function("getLastKltMS",      &GetLastKltMS);
  function("getLastSeedMS",     &GetLastSeedMS);
  function("getNumKeypoints", &GetNumKeypoints);
  function("getTrackState",   &GetTrackState);
  function("getLastTS",       &GetLastTS);
  function("getLastMeanY",  &GetLastMeanY);
  function("getLastProcWH", &GetLastProcWH);
}
