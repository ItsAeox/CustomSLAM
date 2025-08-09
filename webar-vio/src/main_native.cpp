#include <iostream>
#include <opencv2/opencv.hpp>
#include "system.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: vio_app <video or camera_index>\n";
        return 0;
    }

    cv::VideoCapture cap;
    std::string a = argv[1];

    bool opened = false;
    if (std::all_of(a.begin(), a.end(), ::isdigit)) {
        int index = std::stoi(a);
        // Try V4L2 first, then fallback
        cap.open(index, cv::CAP_V4L2);
        if (!cap.isOpened()) cap.open(index, cv::CAP_ANY);
            opened = cap.isOpened();
    } else {
        cap.open(a, cv::CAP_ANY);
        opened = cap.isOpened();
    }

    if (!opened) { std::cerr << "Failed to open source\n"; return -1; }

    // Try to set a sane resolution (optional)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cerr << "Camera opened at " << w << "x" << h << "\n";

    double fx = 0.9*w, fy=0.9*h, cx=w/2.0, cy=h/2.0; // rough guess; replace with real intrinsics

    System sys(Camera(fx,fy,cx,cy,w,h));

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Empty frame, retrying...\n";
            continue;
        }
        cv::Mat rgba;
        cv::cvtColor(frame, rgba, cv::COLOR_BGR2RGBA);
        sys.ProcessFrame(rgba.ptr<uint8_t>(), (double)cv::getTickCount()/cv::getTickFrequency(),
                        rgba.step, /*isRGBA=*/true);

        auto pose = sys.CurrentPoseGL();
        // draw a tiny HUD
        cv::putText(frame, "Inliers: " + std::to_string(sys.lastInliers()), {10,30},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        cv::imshow("VO", frame);
        if (cv::waitKey(1)==27) break;
    }
    return 0;
}
