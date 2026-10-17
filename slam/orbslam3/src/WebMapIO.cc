// ORBSLAM3_WEB (web port, not upstream): room map save and reload. One map (keyframes, map points, covisibility and
// spanning tree, BoW) is written to a portable binary stream and later loaded as the current map of a fresh system;
// tracking then relocalises into it (Tracking::WebAdoptLoadedMap) so AR anchors placed in the earlier session keep
// their coordinates.
//
// Upstream ORB-SLAM3 saves the Atlas with Boost.Serialization archives (System::SaveAtlas), which need the compiled
// Boost library and OpenSSL, absent from the web build. This file reuses upstream's per-object logic unchanged:
// PreSave (pointers -> ids), the serialize() templates of KeyFrame, MapPoint, Map and the IMU types (what is stored),
// and PostLoad (ids -> pointers, covisibility order, keyframe database). Only the archive is ours: a small binary
// stream with the operator& interface the serialize() templates expect. Integers are written at a fixed width
// independent of the platform, so a file saved natively loads in the browser (wasm32 has 32-bit long and size_t):
// version 1 = 64 bits each; version 2 (written now) = zigzag LEB128 varints, as most values are small (octaves,
// indices, ids): ~210 KB per keyframe in version 1, ~120 KB in version 2 (MH01). Not stored, taken from the loading system: the cameras
// (checked against the saved parameters) and the vocabulary (checked by a tag).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "Atlas.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "KeyFrameDatabase.h"
#include "Map.h"
#include "MapPoint.h"
#include "CameraModels/GeometricCamera.h"

namespace ORB_SLAM3
{

namespace
{

const char kMagic[8] = {'W', 'E', 'B', 'V', 'I', 'O', 'M', 'P'};
const int64_t kVersion = 2;  // 1: 64-bit integers (still read); 2: varints

class WebOArchive
{
public:
    typedef std::true_type is_saving;
    typedef std::false_type is_loading;
    explicit WebOArchive(std::ostream& os) : mOs(os) {}
    template<class T> WebOArchive& operator&(T& t) { Save(t); return *this; }
    template<class T> WebOArchive& operator&(const boost::serialization::array_wrapper<T>& a)
    {
        mOs.write(reinterpret_cast<const char*>(a.address()), sizeof(T) * a.count());
        return *this;
    }
    bool Good() const { return mOs.good(); }

private:
    template<class P> void Put(P v) { mOs.write(reinterpret_cast<const char*>(&v), sizeof(P)); }
    template<class T> void Save(T& t)
    {
        if constexpr(std::is_same<T, bool>::value)
            Put<uint8_t>(t ? 1 : 0);
        else if constexpr(std::is_integral<T>::value || std::is_enum<T>::value)
            PutInt(static_cast<int64_t>(t));
        else if constexpr(std::is_floating_point<T>::value)
            Put<T>(t);
        else
            boost::serialization::access::serialize(*this, t, 0);
    }
    void PutInt(int64_t v)  // zigzag LEB128
    {
        uint64_t z = (static_cast<uint64_t>(v) << 1) ^ static_cast<uint64_t>(v >> 63);
        while(z >= 0x80) { mOs.put(static_cast<char>(z | 0x80)); z >>= 7; }
        mOs.put(static_cast<char>(z));
    }
    void Save(std::string& s) { PutInt(s.size()); mOs.write(s.data(), s.size()); }
    template<class T, class A> void Save(std::vector<T, A>& v)
    {
        PutInt(v.size());
        for(T& e : v)
            Save(e);
    }
    template<class A> void Save(std::vector<bool, A>& v)
    {
        PutInt(v.size());
        for(bool b : v)
            Put<uint8_t>(b ? 1 : 0);
    }
    template<class K, class V, class C, class A> void Save(std::map<K, V, C, A>& m)
    {
        PutInt(m.size());
        for(auto& kv : m)
        {
            K k = kv.first;
            Save(k);
            Save(kv.second);
        }
    }
    void Save(DBoW2::BowVector& b) { Save(static_cast<std::map<DBoW2::WordId, DBoW2::WordValue>&>(b)); }
    void Save(DBoW2::FeatureVector& f) { Save(static_cast<std::map<DBoW2::NodeId, std::vector<unsigned int>>&>(f)); }
    // objects owned through pointers (Map::mvpBackupKeyFrames / mvpBackupMapPoints): stored inline
    template<class T> void Save(T*& p)
    {
        Put<uint8_t>(p ? 1 : 0);
        if(p)
            Save(*p);
    }
    std::ostream& mOs;
};

class WebIArchive
{
public:
    typedef std::false_type is_saving;
    typedef std::true_type is_loading;
    explicit WebIArchive(std::istream& is) : mIs(is) {}
    bool mVarint = true;  // false: a version-1 file (64-bit integers)
    template<class T> WebIArchive& operator&(T& t) { Load(t); return *this; }
    template<class T> WebIArchive& operator&(const boost::serialization::array_wrapper<T>& a)
    {
        mIs.read(reinterpret_cast<char*>(a.address()), sizeof(T) * a.count());
        return *this;
    }
    bool Good() const { return mIs.good(); }

private:
    template<class P> P Get()
    {
        P v{};
        mIs.read(reinterpret_cast<char*>(&v), sizeof(P));
        return v;
    }
    // element counts: a corrupt or truncated file must not allocate gigabytes
    size_t Count()
    {
        const int64_t n = GetInt();
        if(!mIs.good() || n < 0 || n > (int64_t(1) << 28))
        {
            mIs.setstate(std::ios::failbit);
            return 0;
        }
        return static_cast<size_t>(n);
    }
    int64_t GetInt()
    {
        if(!mVarint)
            return Get<int64_t>();
        uint64_t z = 0;
        for(int shift = 0; shift < 64; shift += 7)
        {
            const int c = mIs.get();
            if(c == EOF)
            {
                mIs.setstate(std::ios::failbit);
                return 0;
            }
            z |= static_cast<uint64_t>(c & 0x7f) << shift;
            if(!(c & 0x80))
                break;
        }
        return static_cast<int64_t>(z >> 1) ^ -static_cast<int64_t>(z & 1);
    }
    template<class T> void Load(T& t)
    {
        if constexpr(std::is_same<T, bool>::value)
            t = Get<uint8_t>() != 0;
        else if constexpr(std::is_integral<T>::value || std::is_enum<T>::value)
            t = static_cast<T>(GetInt());
        else if constexpr(std::is_floating_point<T>::value)
            t = Get<T>();
        else
            boost::serialization::access::serialize(*this, t, 0);
    }
    void Load(std::string& s)
    {
        s.resize(Count());
        mIs.read(&s[0], s.size());
    }
    template<class T, class A> void Load(std::vector<T, A>& v)
    {
        const size_t n = Count();
        v.clear();
        v.resize(n);
        for(size_t i = 0; i < n && mIs.good(); i++)
            Load(v[i]);
    }
    template<class A> void Load(std::vector<bool, A>& v)
    {
        const size_t n = Count();
        v.assign(n, false);
        for(size_t i = 0; i < n && mIs.good(); i++)
            v[i] = Get<uint8_t>() != 0;
    }
    template<class K, class V, class C, class A> void Load(std::map<K, V, C, A>& m)
    {
        const size_t n = Count();
        m.clear();
        for(size_t i = 0; i < n && mIs.good(); i++)
        {
            K k;
            Load(k);
            Load(m[k]);
        }
    }
    void Load(DBoW2::BowVector& b) { Load(static_cast<std::map<DBoW2::WordId, DBoW2::WordValue>&>(b)); }
    void Load(DBoW2::FeatureVector& f) { Load(static_cast<std::map<DBoW2::NodeId, std::vector<unsigned int>>&>(f)); }
    template<class T> void Load(T*& p)
    {
        p = nullptr;
        if(Get<uint8_t>() && mIs.good())
        {
            p = new T();
            Load(*p);
        }
    }
    std::istream& mIs;
};

}  // namespace

// The vocabulary a map was built with: its BoW vectors and the keyframe database are only valid with the same one
std::string Atlas::WebVocabularyTag(ORBVocabulary* pVoc)
{
    uint64_t h = 1469598103934665603ull;  // FNV-1a over the first 256 word descriptors
    const unsigned int n = std::min(pVoc->size(), 256u);
    for(unsigned int i = 0; i < n; i++)
    {
        const cv::Mat d = pVoc->getWord(i);
        for(size_t b = 0; b < d.total() * d.elemSize(); b++)
            h = (h ^ d.data[b]) * 1099511628211ull;
    }
    char buf[96];
    std::snprintf(buf, sizeof(buf), "k%d L%d words%u fnv%016llx", pVoc->getBranchingFactor(), pVoc->getDepthLevels(),
                  pVoc->size(), static_cast<unsigned long long>(h));
    return buf;
}

bool Atlas::WebSaveMap(std::ostream& os, Map* pMap, std::string& err)
{
    std::unique_lock<std::mutex> lock(mMutexAtlas);
    if(!pMap || pMap->KeyFramesInMap() == 0 || mvpCameras.empty())
    {
        err = "no map to save";
        return false;
    }
    std::set<GeometricCamera*> spCams(mvpCameras.begin(), mvpCameras.end());
    pMap->PreSave(spCams);  // upstream: pointers -> ids (keyframes, map points, graph, cameras)

    WebOArchive ar(os);
    os.write(kMagic, sizeof(kMagic));
    const int64_t version = kVersion;  // raw 8 bytes (as in version 1), then varints
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
    std::string vocTag = WebVocabularyTag(mpORBVocabulary);
    unsigned int camId = mvpCameras[0]->GetId();
    std::vector<float> camParams;
    for(size_t i = 0; i < mvpCameras[0]->size(); i++)
        camParams.push_back(mvpCameras[0]->getParameter(i));
    ar & vocTag & camId & camParams;
    ar & *pMap;  // Map::serialize, with every keyframe and map point inline
    if(!ar.Good())
    {
        err = "write failed";
        return false;
    }
    return true;
}

Map* Atlas::WebLoadMap(std::istream& is, std::string& err)
{
    char magic[sizeof(kMagic)] = {};
    is.read(magic, sizeof(magic));
    if(!is.good() || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
    {
        err = "not a WebVIO map file";
        return nullptr;
    }
    WebIArchive ar(is);
    int64_t version = 0;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    std::string vocTag;
    unsigned int camId = 0;
    std::vector<float> camParams;
    if(version != 1 && version != 2)
    {
        err = "unsupported map file version";
        return nullptr;
    }
    ar.mVarint = version >= 2;
    ar & vocTag & camId & camParams;
    if(!ar.Good())
    {
        err = "truncated map file header";
        return nullptr;
    }
    if(vocTag != WebVocabularyTag(mpORBVocabulary))
    {
        err = "the map was built with another vocabulary (" + vocTag + ")";
        return nullptr;
    }
    std::unique_lock<std::mutex> lock(mMutexAtlas);
    if(mvpCameras.empty() || mvpCameras[0]->size() != camParams.size())
    {
        err = "camera model differs from the saved map";
        return nullptr;
    }
    for(size_t i = 0; i < camParams.size(); i++)
    {
        const float a = camParams[i], b = mvpCameras[0]->getParameter(i);
        if(std::fabs(a - b) > 0.01f * std::max(1.f, std::fabs(a)))  // 1%: the saved keyframes keep their own fx, fy...
        {
            err = "camera calibration differs from the saved map";
            return nullptr;
        }
    }

    Map* pMap = new Map();
    ar & *pMap;
    if(!ar.Good())
    {
        err = "truncated or corrupt map file";
        return nullptr;  // partly read objects leak: a failed load is followed by a restart
    }

    // upstream PostLoad: ids -> pointers, best covisibles, keyframes into the (empty) keyframe database
    std::map<unsigned int, GeometricCamera*> mpCams;
    mpCams[camId] = mvpCameras[0];
    pMap->PostLoad(mpKeyFrameDB, mpORBVocabulary, mpCams);

    // the loaded map replaces the empty first map of the fresh system and becomes the current map
    if(mpCurrentMap && mpCurrentMap->KeyFramesInMap() == 0)
    {
        mspMaps.erase(mpCurrentMap);
        mpCurrentMap->SetBad();
        mspBadMaps.insert(mpCurrentMap);
    }
    else if(mpCurrentMap)
        mpCurrentMap->SetStoredMap();
    mspMaps.insert(pMap);
    pMap->SetCurrentMap();
    mpCurrentMap = pMap;

    // new objects of this session get ids after the loaded ones (markers such as mnTrackReferenceForFrame and
    // mnLastFrameSeen compare ids, so the frame counter also continues after the loaded keyframes' frames)
    long unsigned int maxKF = 0, maxFrame = 0, maxMP = 0;
    for(KeyFrame* pKF : pMap->GetAllKeyFrames())
    {
        maxKF = std::max(maxKF, pKF->mnId);
        maxFrame = std::max(maxFrame, pKF->mnFrameId);
    }
    for(MapPoint* pMP : pMap->GetAllMapPoints())
        maxMP = std::max(maxMP, pMP->mnId);
    KeyFrame::nNextId = std::max(KeyFrame::nNextId, maxKF + 1);
    MapPoint::nNextId = std::max(MapPoint::nNextId, maxMP + 1);
    Frame::nNextId = std::max(Frame::nNextId, maxFrame + 1);
    Map::nNextId = std::max(Map::nNextId, pMap->GetId() + 1);
    mnLastInitKFidMap = std::max(mnLastInitKFidMap, maxKF + 1);
    return pMap;
}

}  // namespace ORB_SLAM3
