// JPEG (or any OpenCV-readable) frame -> 8-bit gray, for the AlvaAR driver on JPEG datasets (the synthetic set, phone
// recordings). It is the call WebVIO's runner makes (orbslam3_run.cc readGray: cv::imdecode(buf, IMREAD_GRAYSCALE)) with
// the same OpenCV 4.10.0 WebAssembly build (opencv-4.10.0-wasm-mt), so both systems get the same pixels. Built by build_jpeg_gray.sh.
#include <emscripten/emscripten.h>
#include <opencv2/imgcodecs.hpp>
#include <cstring>

extern "C" {
// Decodes n bytes at buf into out (capacity cap); wh[0], wh[1] = width, height. Returns 0, or -1 (decode failed) /
// -2 (out too small).
EMSCRIPTEN_KEEPALIVE int decode_gray(const unsigned char* buf, int n, unsigned char* out, int cap, int* wh) {
  cv::Mat im = cv::imdecode(cv::Mat(1, n, CV_8U, const_cast<unsigned char*>(buf)), cv::IMREAD_GRAYSCALE);
  if (im.empty()) return -1;
  if (im.total() > (size_t)cap) return -2;
  wh[0] = im.cols; wh[1] = im.rows;
  for (int r = 0; r < im.rows; ++r) std::memcpy(out + (size_t)r * im.cols, im.ptr(r), im.cols);
  return 0;
}
}
