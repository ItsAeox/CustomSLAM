#pragma once
#include <Eigen/Core>

struct Camera {
    int width=0, height=0;
    double fx=0, fy=0, cx=0, cy=0;

    Camera() = default;
    Camera(double fx_, double fy_, double cx_, double cy_, int w_, int h_)
        : width(w_), height(h_), fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

  Eigen::Matrix3d K() const {
        Eigen::Matrix3d k; k.setZero();
        k(0,0)=fx; k(1,1)=fy; k(0,2)=cx; k(1,2)=cy; k(2,2)=1.0;
        return k;
    }
};
