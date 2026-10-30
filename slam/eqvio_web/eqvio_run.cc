// Replay of a sequence in the EuRoC / phone-recording layout (mav0/cam0/data.csv + data/<file>, mav0/imu0/data.csv) through
// EqvioWeb, natively or under Node (WASM, NODERAWFS): the parity and cost check of the browser build against eqvio_opt.
// Camera: <sequence>/settings.yaml (Camera1.fx/fy/cx/cy, Camera.width/height, IMU.T_b_c1).
// Writes <out>/traj_camclock.txt (camera pose, TUM, keyed by the frame's camera stamp, once initialised) and
// <out>/timing.csv (frame, features, GIFT ms, filter ms, whole frame ms excluding the image read).
//   eqvio_run <sequence dir> <EqVIO config.yaml> <out dir> [--color]   (--color: BGR frames as eqvio_opt; default grey)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>

#include "EqvioWeb.h"

namespace {
struct Row {
    long long ns;
    std::vector<std::string> cols;
};

std::vector<Row> readCsv(const std::string& path) {
    std::vector<Row> rows;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        Row r;
        std::stringstream ss(line);
        std::string c;
        while (std::getline(ss, c, ',')) r.cols.push_back(c);
        r.ns = std::stoll(r.cols[0]);
        rows.push_back(r);
    }
    return rows;
}

std::string readFile(const std::string& path) {
    std::ifstream f(path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

double yamlNumber(const std::string& txt, const std::string& key) {  // "key: value" of an OpenCV settings.yaml
    const size_t p = txt.find("\n" + key + ":");
    return p == std::string::npos ? NAN : std::stod(txt.substr(txt.find(':', p) + 1));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: eqvio_run <sequence dir> <EqVIO config.yaml> <out dir> [--color]\n";
        return 1;
    }
    const std::string seq = argv[1], out = argv[3];
    const bool color = argc > 4 && std::string(argv[4]) == "--color";
    const std::string st = readFile(seq + "/settings.yaml");
    const std::vector<double> K = {yamlNumber(st, "Camera1.fx"), yamlNumber(st, "Camera1.fy"), yamlNumber(st, "Camera1.cx"),
                                   yamlNumber(st, "Camera1.cy")};
    const int w = (int)yamlNumber(st, "Camera.width"), h = (int)yamlNumber(st, "Camera.height");
    std::vector<double> Tbc;
    {
        const size_t p0 = st.find('[', st.find("data:", st.find("IMU.T_b_c1")));
        std::stringstream ts(st.substr(p0 + 1, st.find(']', p0) - p0 - 1));
        std::string c;
        while (std::getline(ts, c, ',')) Tbc.push_back(std::stod(c));
    }
    EqvioWeb vio(readFile(argv[2]), K, w, h, Tbc);
    const auto imu = readCsv(seq + "/mav0/imu0/data.csv");
    const auto cam = readCsv(seq + "/mav0/cam0/data.csv");
    std::ofstream traj(out + "/traj_camclock.txt"), timing(out + "/timing.csv");
    traj << "# t tx ty tz qx qy qz qw (camera, EqvioWeb, camera clock)\n";
    timing << "frame,features,features_ms,filter_ms,frame_ms\n";
    size_t i = 0;
    for (size_t k = 0; k < cam.size(); k++) {
        const double t = cam[k].ns * 1e-9;
        for (; i < imu.size() && imu[i].ns * 1e-9 < t - vio.cameraLag; i++)
            vio.addImu(imu[i].ns * 1e-9, Eigen::Vector3d(std::stod(imu[i].cols[1]), std::stod(imu[i].cols[2]), std::stod(imu[i].cols[3])),
                       Eigen::Vector3d(std::stod(imu[i].cols[4]), std::stod(imu[i].cols[5]), std::stod(imu[i].cols[6])));
        const cv::Mat img = cv::imread(seq + "/mav0/cam0/data/" + cam[k].cols[1], color ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        const auto t0 = std::chrono::steady_clock::now();
        vio.addFrame(t, img);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        timing << k << "," << vio.nFeatures << "," << vio.msFeatures << "," << vio.msFilter << "," << ms << "\n";
        if (!vio.initialised()) continue;
        const Eigen::Matrix4d T = vio.Twc();
        const Eigen::Quaterniond q(Eigen::Matrix3d(T.block<3, 3>(0, 0)));
        char buf[256];
        std::snprintf(buf, sizeof buf, "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", t, T(0, 3), T(1, 3), T(2, 3), q.x(), q.y(),
                      q.z(), q.w());
        traj << buf;
    }
    std::cout << out << ": " << cam.size() << " frames, " << imu.size() << " IMU samples\n";
    return 0;
}
