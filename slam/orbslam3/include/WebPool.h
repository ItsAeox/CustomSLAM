// ORBSLAM3_WEB (web port, not upstream): one persistent pool of helper threads for work the tracking thread splits up
// (ORB pyramid levels: Web.ORB.Parallel; optical-flow points: Web.KLT.Parallel). Own threads because OpenCV's WASM build
// has no parallel framework (cv::parallel_for_ runs serially there; natively OpenCV spreads LK over every core). One pool
// for both, so the browser's pre-spawned workers are not exhausted (10; 6 deadlocked, 2026-10-17). Used by the tracking
// thread only, never nested: the caller works too, and Run returns when every task has finished.
#ifndef ORBSLAM3_WEBPOOL_H
#define ORBSLAM3_WEBPOOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

namespace ORB_SLAM3
{

class WebPool
{
public:
    explicit WebPool(int nHelpers)
    {
        for(int i = 0; i < nHelpers; i++)
            std::thread([this] { Loop(); }).detach();  // never joined: the pool lives until the process ends
    }
    int Threads() const { return mnThreads; }
    void SetThreads(int n) { mnThreads = n; }
    // runs f(0..nTasks-1) on the helpers and the calling thread; returns when all have finished
    void Run(int nTasks, const std::function<void(int)>& f)
    {
        uint32_t gen;
        {
            std::unique_lock<std::mutex> lock(mMutex);
            gen = ++mnGeneration;
            mpTask = &f;
            mnTasks = nTasks;
            mnDone = 0;
            mnNext.store((uint64_t)gen << 32);
        }
        mCvWork.notify_all();
        Work(f, nTasks, gen);
        std::unique_lock<std::mutex> lock(mMutex);
        mCvDone.wait(lock, [&] { return mnDone == mnTasks; });
    }

private:
    // next task = generation (high 32 bits) + index: a helper that woke for an earlier call cannot take a task
    void Work(const std::function<void(int)>& f, int nTasks, uint32_t gen)
    {
        while(true)
        {
            uint64_t v = mnNext.load();
            if((uint32_t)(v >> 32) != gen || (int)(uint32_t)v >= nTasks)
                return;
            if(!mnNext.compare_exchange_weak(v, v + 1))
                continue;
            f((int)(uint32_t)v);
            std::unique_lock<std::mutex> lock(mMutex);
            if(++mnDone == mnTasks)
                mCvDone.notify_all();
        }
    }
    void Loop()
    {
        uint32_t seen = 0;
        while(true)
        {
            const std::function<void(int)>* f;
            int n;
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCvWork.wait(lock, [&] { return mnGeneration != seen; });
                seen = mnGeneration;
                f = mpTask;
                n = mnTasks;
            }
            Work(*f, n, seen);
        }
    }
    std::mutex mMutex;
    std::condition_variable mCvWork, mCvDone;
    const std::function<void(int)>* mpTask = nullptr;
    int mnTasks = 0, mnDone = 0, mnThreads = 1;
    std::atomic<uint64_t> mnNext{0};
    uint32_t mnGeneration = 0;
};

// The shared pool: 4 threads (3 helpers + the caller), created on first use and never destroyed (a static destructor
// under the waiting helpers hung native/Node processes at exit, 2026-10-17).
inline WebPool& webPool()
{
    static WebPool* pool = []() { WebPool* p = new WebPool(3); p->SetThreads(4); return p; }();
    return *pool;
}

}  // namespace ORB_SLAM3

#endif
