// slam/imu_initializer.cc
#include "slam/imu_initializer.h"
#include <sophus/se3.hpp>


struct GravityResidual {
    GravityResidual(const Eigen::Vector3d& a_meas, double dt,
                    const Eigen::Vector3d& t0, const Eigen::Vector3d& t1)
        : a_meas_(a_meas), dt_(dt), t0_(t0), t1_(t1) {}

    template<typename T>
    bool operator()(const T* const scale,
                    const T* const gvec,
                    T* residuals) const {
        Eigen::Matrix<T,3,1> t0T = t0_.cast<T>();
        Eigen::Matrix<T,3,1> t1T = t1_.cast<T>();
        Eigen::Matrix<T,3,1> acc_meas_T = a_meas_.cast<T>();

        // Acceleration due to motion: (s * (t1 - t0)) / dt^2
        Eigen::Matrix<T,3,1> a_model = (*scale) * (t1T - t0T) / (T(dt_ * dt_));
        Eigen::Matrix<T,3,1> gT(gvec[0], gvec[1], gvec[2]);

        Eigen::Matrix<T,3,1> pred = a_model + gT;
        residuals[0] = pred[0] - acc_meas_T[0];
        residuals[1] = pred[1] - acc_meas_T[1];
        residuals[2] = pred[2] - acc_meas_T[2];
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& a_meas, double dt,
                                       const Eigen::Vector3d& t0,
                                       const Eigen::Vector3d& t1) {
        return new ceres::AutoDiffCostFunction<GravityResidual, 3, 1, 3>(
            new GravityResidual(a_meas, dt, t0, t1));
    }

    const Eigen::Vector3d a_meas_;
    const double dt_;
    const Eigen::Vector3d t0_, t1_;
};

ScaleInitResult ScaleInitializer::EstimateScaleAndGravity(
    const Sophus::SE3d& T0_wc,
    const Sophus::SE3d& T1_wc,
    const std::vector<IMUSample>& imu_span) {

    if (imu_span.size() < 5) return {};

    const Eigen::Vector3d t0 = T0_wc.translation();
    const Eigen::Vector3d t1 = T1_wc.translation();
    const double t_start = imu_span.front().t;
    const double t_end   = imu_span.back().t;
    const double total_dt = t_end - t_start;
    if (total_dt <= 0.01) return {};

    double scale = 1.0;
    Eigen::Vector3d gvec(0, -9.81, 0);

    ceres::Problem problem;
    problem.AddParameterBlock(&scale, 1);
    problem.AddParameterBlock(gvec.data(), 3);

    for (size_t i = 1; i < imu_span.size(); ++i) {
        const auto& s0 = imu_span[i-1];
        const auto& s1 = imu_span[i];
        const double dt = s1.t - s0.t;
        if (dt <= 0.002 || dt > 0.05) continue;

        Eigen::Vector3d a_avg = 0.5 * (s0.acc + s1.acc);
        ceres::CostFunction* cost = GravityResidual::Create(a_avg, total_dt, t0, t1);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(0.5), &scale, gvec.data());
    }

    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = 25;
    opts.num_threads = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    ScaleInitResult result;
    result.scale = scale;
    result.gravity = gvec;
    result.success = summary.termination_type == ceres::CONVERGENCE;
    return result;
}
