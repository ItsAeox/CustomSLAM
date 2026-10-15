// BEBLID / TEBLID binary descriptors, vendored from opencv_contrib 4.10.0 (modules/xfeatures2d/src/beblid.cpp and the
// *.p256/p512.hpp weak-learner tables), Apache-2.0 (LICENSE.opencv_contrib). Author of the original implementation:
// Iago Suarez. Papers: BEBLID (Pattern Recognition Letters 2020); TEBLID / BAD ("Revisiting Binary Local Image
// Description for Resource Limited Devices", RA-L 2021).
// Only change for the web port: these class declarations (copied from opencv2/xfeatures2d.hpp) replace contrib's
// precomp.hpp, so the descriptor builds against plain OpenCV core/imgproc/features2d without the contrib module.
#ifndef WEB_TEBLID_HPP
#define WEB_TEBLID_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace cv
{
namespace xfeatures2d
{

class CV_EXPORTS_W BEBLID : public Feature2D
{
public:
    enum BeblidSize
    {
        SIZE_512_BITS = 100, SIZE_256_BITS = 101,
    };
    CV_WRAP static Ptr<BEBLID> create(float scale_factor, int n_bits = BEBLID::SIZE_512_BITS);
    CV_WRAP virtual void setScaleFactor(float scale_factor) = 0;
    CV_WRAP virtual float getScaleFactor() const = 0;
    CV_WRAP String getDefaultName() const CV_OVERRIDE;
};

// scale_factor 1.00 for ORB keypoints. TEBLID::SIZE_256_BITS gives 32-byte descriptors, like ORB.
class CV_EXPORTS_W TEBLID : public Feature2D
{
public:
    enum TeblidSize
    {
        SIZE_256_BITS = 102, SIZE_512_BITS = 103,
    };
    CV_WRAP static Ptr<TEBLID> create(float scale_factor, int n_bits = TEBLID::SIZE_256_BITS);
    CV_WRAP String getDefaultName() const CV_OVERRIDE;
};

}  // namespace xfeatures2d
}  // namespace cv

#endif
