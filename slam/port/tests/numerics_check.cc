// Deterministic numerics check: runs ORB-SLAM3 building blocks on identical, self-generated inputs and prints the
// results with full precision, so the native and WebAssembly builds can be diffed component by component.
//
// Usage: numerics_check [<ORBvoc.bin>] [--verbose]     (no vocabulary: skip the BoW test; --verbose: g2o iterations)
//
// All inputs come from std::mt19937 (standard-defined) and a hand-written Box-Muller transform (no std::*_distribution,
// whose implementations differ between libstdc++ and libc++).

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include "CameraModels/KannalaBrandt8.h"
#include "CameraModels/Pinhole.h"
#include "OptimizableTypes.h"
#include "ORBVocabulary.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"

using namespace ORB_SLAM3;

static std::mt19937 rng(12345);
static bool gVerbose = false;  // --verbose (environment variables don't reach WASM programs)
static double uni() { return (double)rng() / 4294967296.0; }               // [0,1)
static double gauss() {                                                     // Box-Muller
  double u1 = uni(), u2 = uni();
  if (u1 < 1e-12) u1 = 1e-12;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
}

// NOTE: one random draw per statement. Argument evaluation order is unspecified in C++ (GCC: right-to-left,
// Clang: left-to-right), so f(rand(), rand()) gives different inputs on different compilers.
static Eigen::Vector2d gauss2() { const double a = gauss(); const double b = gauss(); return Eigen::Vector2d(a, b); }
static Eigen::Vector3d gauss3() { const double a = gauss(); const double b = gauss(); const double c = gauss(); return Eigen::Vector3d(a, b, c); }
static Eigen::Vector3d uni3(double sx, double sy, double z0, double sz) {
  const double a = uni(); const double b = uni(); const double c = uni();
  return Eigen::Vector3d((a - 0.5) * sx, (b - 0.5) * sy, z0 + c * sz);
}

static g2o::SE3Quat perturbed(const g2o::SE3Quat& T, double rot, double trans) {
  Eigen::Matrix<double, 6, 1> d;
  for (int i = 0; i < 3; ++i) d[i] = rot * gauss();
  for (int i = 3; i < 6; ++i) d[i] = trans * gauss();
  return g2o::SE3Quat::exp(d) * T;
}

static void printSE3(const char* tag, const g2o::SE3Quat& T) {
  const auto q = T.rotation().coeffs();  // x y z w
  const auto t = T.translation();
  std::printf("%s q=[%.12f %.12f %.12f %.12f] t=[%.12f %.12f %.12f]\n", tag, q[0], q[1], q[2], q[3], t[0], t[1], t[2]);
}

// ---------------------------------------------------------------- T1: BoW transform
static void testBoW(const char* vocPath) {
  ORBVocabulary voc;
  if (!voc.loadFromBinaryFile(vocPath)) { std::printf("T1 BoW: vocabulary load FAILED\n"); return; }
  std::mt19937 r(7);
  double wsum = 0; unsigned long long idsum = 0, fsum = 0; size_t nwords = 0;
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<cv::Mat> desc;
    for (int i = 0; i < 800; ++i) {
      cv::Mat d(1, 32, CV_8U);
      for (int j = 0; j < 32; ++j) d.at<uint8_t>(0, j) = (uint8_t)(r() & 0xff);
      desc.push_back(d);
    }
    DBoW2::BowVector bv; DBoW2::FeatureVector fv;
    voc.transform(desc, bv, fv, 4);
    for (auto& [w, v] : bv) { idsum += w; wsum += v; ++nwords; }
    for (auto& [node, feats] : fv) for (auto f : feats) fsum += (unsigned long long)node * 1000 + f;
  }
  std::printf("T1 BoW: words=%zu idsum=%llu wsum=%.15f fvsum=%llu\n", nwords, idsum, wsum, fsum);
}

// ---------------------------------------------------------------- T2: pose-only optimisation (as PoseOptimization)
static void testPoseOptimization(GeometricCamera* cam, const char* name, int W, int H) {
  rng.seed(100);
  const g2o::SE3Quat Ttrue = g2o::SE3Quat::exp((Eigen::Matrix<double, 6, 1>() << 0.1, -0.2, 0.05, 0.3, -0.1, 0.2).finished());
  g2o::SparseOptimizer opt;
  auto* ls = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
  opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(new g2o::BlockSolver_6_3(ls)));
  auto* v = new g2o::VertexSE3Expmap();
  v->setEstimate(perturbed(Ttrue, 0.03, 0.05));
  v->setId(0);
  opt.addVertex(v);
  int n = 0;
  while (n < 300) {
    Eigen::Vector3d Xc = uni3(6, 4, 1.0, 8);
    Eigen::Vector2d uv = cam->project(Xc);
    if (uv[0] < 0 || uv[1] < 0 || uv[0] >= W || uv[1] >= H) continue;
    Eigen::Vector3d Xw = Ttrue.inverse().map(Xc);
    const bool outlier = (n % 10 == 0);
    Eigen::Vector2d obs = uv + gauss2() * (outlier ? 40.0 : 1.0);
    auto* e = new EdgeSE3ProjectXYZOnlyPose();
    e->setVertex(0, v);
    e->setMeasurement(obs);
    e->setInformation(Eigen::Matrix2d::Identity());
    auto* rk = new g2o::RobustKernelHuber;
    rk->setDelta(std::sqrt(5.991));
    e->setRobustKernel(rk);
    e->pCamera = cam;
    e->Xw = Xw;
    opt.addEdge(e);
    ++n;
  }
  opt.initializeOptimization(0);
  opt.setVerbose(gVerbose);  // per-iteration chi2 / lambda on stderr
  opt.computeActiveErrors();
  std::printf("T2 initial chi2[%s]=%.12f\n", name, opt.activeRobustChi2());
  opt.optimize(10);
  char tag[64];
  std::snprintf(tag, sizeof tag, "T2 PoseOpt[%s]", name);
  printSE3(tag, v->estimate());
}

// ---------------------------------------------------------------- T3: bundle adjustment (as LocalBundleAdjustment)
// Wraps a g2o linear solver and counts failed solves (LinearSolverEigen returns false when the Cholesky factorisation
// fails; Levenberg-Marquardt then rejects the step and raises lambda).
template <class Solver, class M>
struct CountingSolver : Solver {
  int calls = 0, fails = 0;
  bool solve(const g2o::SparseBlockMatrix<M>& A, double* x, double* b) override {
    ++calls;
    const bool ok = Solver::solve(A, x, b);
    if (!ok) ++fails;
    return ok;
  }
};

// solver: "eigen" (LinearSolverEigen, as ORB-SLAM3's LocalBundleAdjustment), "eigen-block" (with block AMD ordering),
// "dense" (LinearSolverDense)
static void testBA(GeometricCamera* cam, const std::string& solver) {
  rng.seed(200);
  const int nKF = 6, nPts = 250;
  std::vector<g2o::SE3Quat> Tcw(nKF);
  for (int k = 0; k < nKF; ++k)
    Tcw[k] = g2o::SE3Quat::exp((Eigen::Matrix<double, 6, 1>() << 0.02 * k, 0.03 * k, -0.01 * k, -0.15 * k, 0.05 * k, 0.02 * k).finished());
  std::vector<Eigen::Vector3d> Xw(nPts);
  for (auto& X : Xw) X = uni3(6, 4, 2.0, 6);

  g2o::SparseOptimizer opt;
  using PM = g2o::BlockSolver_6_3::PoseMatrixType;
  CountingSolver<g2o::LinearSolverEigen<PM>, PM>* lsE = nullptr;
  CountingSolver<g2o::LinearSolverDense<PM>, PM>* lsD = nullptr;
  g2o::LinearSolver<PM>* ls = nullptr;
  if (solver == "dense") {
    ls = lsD = new CountingSolver<g2o::LinearSolverDense<PM>, PM>();
  } else {
    ls = lsE = new CountingSolver<g2o::LinearSolverEigen<PM>, PM>();
    lsE->setBlockOrdering(solver == "eigen-block");
  }
  opt.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(new g2o::BlockSolver_6_3(ls)));
  for (int k = 0; k < nKF; ++k) {
    auto* v = new g2o::VertexSE3Expmap();
    v->setEstimate(k < 2 ? Tcw[k] : perturbed(Tcw[k], 0.01, 0.02));
    v->setId(k);
    v->setFixed(k < 2);
    opt.addVertex(v);
  }
  for (int i = 0; i < nPts; ++i) {
    auto* vp = new g2o::VertexSBAPointXYZ();
    vp->setEstimate(Xw[i] + gauss3() * 0.05);
    vp->setId(nKF + i);
    vp->setMarginalized(true);
    opt.addVertex(vp);
    for (int k = 0; k < nKF; ++k) {
      Eigen::Vector3d Xc = Tcw[k].map(Xw[i]);
      if (Xc[2] <= 0.1) continue;
      Eigen::Vector2d obs = cam->project(Xc) + gauss2();
      auto* e = new EdgeSE3ProjectXYZ();
      e->setVertex(0, vp);
      e->setVertex(1, opt.vertex(k));
      e->setMeasurement(obs);
      e->setInformation(Eigen::Matrix2d::Identity());
      auto* rk = new g2o::RobustKernelHuber;
      rk->setDelta(std::sqrt(5.991));
      e->setRobustKernel(rk);
      e->pCamera = cam;
      opt.addEdge(e);
    }
  }
  opt.initializeOptimization();
  opt.computeActiveErrors();
  std::printf("T3[%s] initial chi2=%.12f edges=%zu\n", solver.c_str(), opt.activeRobustChi2(), opt.edges().size());
  opt.setVerbose(gVerbose);
  opt.optimize(10);
  for (int k = 2; k < nKF; ++k) {
    char tag[64]; std::snprintf(tag, sizeof tag, "T3[%s] BA KF%d", solver.c_str(), k);
    printSE3(tag, static_cast<g2o::VertexSE3Expmap*>(opt.vertex(k))->estimate());
  }
  double s = 0;
  for (int i = 0; i < nPts; ++i) s += static_cast<g2o::VertexSBAPointXYZ*>(opt.vertex(nKF + i))->estimate().sum();
  const int calls = lsE ? lsE->calls : lsD->calls, fails = lsE ? lsE->fails : lsD->fails;
  std::printf("T3[%s] BA pointsum=%.12f chi2=%.12f linear solves=%d failed=%d\n", solver.c_str(), s,
              opt.activeRobustChi2(), calls, fails);
}

// ---------------------------------------------------------------- T4/T5: camera models, cv::undistortPoints
static void testCameras(GeometricCamera* pin, GeometricCamera* kb) {
  double sp = 0, su = 0, sj = 0, kp = 0, ku = 0, kj = 0;
  for (int y = -10; y <= 10; ++y)
    for (int x = -10; x <= 10; ++x) {
      Eigen::Vector3d X(x * 0.07, y * 0.05, 1.0 + 0.01 * (x + y + 20));
      Eigen::Vector2d a = pin->project(X), b = kb->project(X);
      sp += a.sum(); kp += b.sum();
      sj += pin->projectJac(X).sum(); kj += kb->projectJac(X).sum();
      Eigen::Vector3f ua = pin->unprojectEig(cv::Point2f((float)a[0], (float)a[1]));
      Eigen::Vector3f ub = kb->unprojectEig(cv::Point2f((float)b[0], (float)b[1]));
      su += ua.cast<double>().sum(); ku += ub.cast<double>().sum();
    }
  std::printf("T4 Pinhole proj=%.10f jac=%.10f unproj=%.10f\n", sp, sj, su);
  std::printf("T4 KB8     proj=%.10f jac=%.10f unproj=%.10f\n", kp, kj, ku);

  cv::Mat K = (cv::Mat_<float>(3, 3) << 458.654f, 0, 367.215f, 0, 457.296f, 248.375f, 0, 0, 1);
  cv::Mat D = (cv::Mat_<float>(4, 1) << -0.28340811f, 0.07395907f, 0.00019359f, 1.76187114e-05f);
  cv::Mat pts(1, 0, CV_32FC2);
  std::vector<cv::Point2f> in, out;
  for (int y = 0; y < 480; y += 40) for (int x = 0; x < 752; x += 47) in.emplace_back((float)x, (float)y);
  cv::undistortPoints(in, out, K, D, cv::Mat(), K);  // as Frame::UndistortKeyPoints
  double s = 0; for (auto& p : out) s += p.x * 1.0 + p.y * 3.0;
  std::printf("T5 undistortPoints sum=%.10f first=(%.8f,%.8f)\n", s, out[0].x, out[0].y);
}

// ---------------------------------------------------------------- T6: Sophus
static void testSophus() {
  Sophus::SE3f T = Sophus::SE3f::exp((Sophus::SE3f::Tangent() << 0.3f, -0.2f, 0.1f, 0.4f, 0.5f, -0.6f).finished());
  Sophus::SE3f U = T * T.inverse() * T;
  auto l = U.log();
  std::printf("T6 Sophus log=[%.9f %.9f %.9f %.9f %.9f %.9f]\n", l[0], l[1], l[2], l[3], l[4], l[5]);
}

int main(int argc, char** argv) {
  Pinhole pin(std::vector<float>{458.654f, 457.296f, 367.215f, 248.375f});
  KannalaBrandt8 kb(std::vector<float>{190.978477f, 190.973307f, 254.931706f, 256.897442f,
                                       0.003482389402f, 0.000715034845f, -0.002053236141f, 0.000202936736f});
  for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == "--verbose") gVerbose = true;
  if (argc > 1 && std::string(argv[1]) != "--verbose") testBoW(argv[1]);
  testPoseOptimization(&pin, "pinhole", 752, 480);
  testPoseOptimization(&kb, "kb8", 512, 512);
  for (const char* solver : {"eigen", "eigen-block", "dense"}) testBA(&pin, solver);
  testCameras(&pin, &kb);
  testSophus();
  return 0;
}
