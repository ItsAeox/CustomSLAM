// ORBSLAM3_WEB (web port, not upstream): runtime knobs for the browser build, read from optional "Web.*" keys in the
// settings file when the System starts. Every default reproduces stock ORB-SLAM3 exactly, so a settings file without
// Web.* keys runs the unmodified algorithm (used as the baseline in every comparison).
#ifndef ORBSLAM3_WEBCONFIG_H
#define ORBSLAM3_WEBCONFIG_H

#include <iostream>
#include <string>

#include <opencv2/core/persistence.hpp>

namespace ORB_SLAM3
{

struct WebConfig
{
    int lbaMaxOptKFs = 0;      // Web.LBA.MaxOptKFs: optimised keyframes in local BA, best-covisible first (0 = all, stock)
    int lbaMaxFixedKFs = 0;    // Web.LBA.MaxFixedKFs: fixed keyframes, most observations of local points first (0 = all, stock)
    int lbaIterations = 10;    // Web.LBA.Iterations: Levenberg-Marquardt iterations of local BA (stock 10)
    int mappingNeighbors = 0;  // Web.Mapping.Neighbors: covisible keyframes for triangulation / fusion (0 = stock: 30 mono)
};

inline WebConfig& webConfig()
{
    static WebConfig cfg;
    return cfg;
}

inline void loadWebConfig(const std::string& settingsFile)
{
    WebConfig& c = webConfig();
    c = WebConfig();
    cv::FileStorage fs(settingsFile, cv::FileStorage::READ);
    if (!fs.isOpened()) return;
    auto readInt = [&fs](const char* key, int& v) {
        cv::FileNode n = fs[key];
        if (!n.empty() && (n.isInt() || n.isReal())) v = (int)n.real();
    };
    readInt("Web.LBA.MaxOptKFs", c.lbaMaxOptKFs);
    readInt("Web.LBA.MaxFixedKFs", c.lbaMaxFixedKFs);
    readInt("Web.LBA.Iterations", c.lbaIterations);
    readInt("Web.Mapping.Neighbors", c.mappingNeighbors);
    std::cout << "Web config: LBA.MaxOptKFs " << c.lbaMaxOptKFs << ", LBA.MaxFixedKFs " << c.lbaMaxFixedKFs
              << ", LBA.Iterations " << c.lbaIterations << ", Mapping.Neighbors " << c.mappingNeighbors
              << " (0 = stock)" << std::endl;
}

}  // namespace ORB_SLAM3

#endif
