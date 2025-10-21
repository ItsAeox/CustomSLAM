#include <emscripten/bind.h>
#include "system.h"

using namespace emscripten;

static System gSys;


// --- wrappers so embind gets raw function pointers ---
static int    GetNumKeypoints() { return gSys.getNumKeypoints(); }
static int    GetTrackState()   { return gSys.getTrackState(); }
static double GetLastTS()       { return gSys.getLastTS(); }

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

EMSCRIPTEN_BINDINGS(vio_bindings_pointtrack) {
  function("initSystem",   &initSystem);
  function("feedFrameJS",  &feedFrameJS);
  function("getPoints2DArray", &getPoints2D);       // old JS array builder
  function("getPoints2D",      &getPoints2D_Typed); // fast typed view


  // use wrappers (no lambdas)
  function("getNumKeypoints", &GetNumKeypoints);
  function("getTrackState",   &GetTrackState);
  function("getLastTS",       &GetLastTS);

}
