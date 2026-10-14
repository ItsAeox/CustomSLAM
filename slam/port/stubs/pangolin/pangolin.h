// Stand-in for <pangolin/pangolin.h> in the web build (Pangolin/OpenGL viewer is not compiled).
// ORB-SLAM3 headers only need the OpenGlMatrix type in MapDrawer's (unused) drawing API.
#pragma once

// Pangolin pulls in the OpenGL headers; Map.h uses GLubyte for an (unused) thumbnail buffer.
typedef unsigned char GLubyte;

namespace pangolin {

struct OpenGlMatrix {
  double m[16];
  OpenGlMatrix() { SetIdentity(); }
  void SetIdentity() {
    for (int i = 0; i < 16; ++i) m[i] = (i % 5 == 0) ? 1.0 : 0.0;
  }
};

}  // namespace pangolin
