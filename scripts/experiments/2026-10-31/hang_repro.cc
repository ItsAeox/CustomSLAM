// ISSUES G23 / G33 (replays stopped at ~0 % CPU, WASM only, 2 in 630 runs on 2 Nov, both while tracking was lost):
// a small program with orbslam3_run's thread layout, to try to hit the hang in minutes. In WASM orbslam3_run runs 7
// threads for the whole run (main = PROXY_TO_PTHREAD, frame source, LocalMapping, LoopClosing, 3 WebPool helpers) and the
// pool has 8 workers; every two-view try of the map start (TwoViewReconstruction::Reconstruct) starts and joins 2 more
// threads (homography + fundamental), so while tracking is lost the map start creates ~2 threads per frame. Thread start
// and exit go through Node's main thread (spawnThread / cleanupThread), which also serves every write to a file.
// Per iteration, on the "tracking" thread: a WebPool run (KLT), a line to stdout and to a trace file, then the two-view
// threads: mode 0 = 2 new threads joined (as Reconstruct), mode 1 = the same work on the pool. Other threads: frame
// source (frame buffers, sleeps), LocalMapping (malloc / free churn, a line now and then), LoopClosing (sleeps).
// Usage: hang_repro.js <iterations> <mode 0|1> <trace file>   (prints "done" at the end)
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// as slam/orbslam3/include/WebPool.h
class WebPool
{
public:
    explicit WebPool(int nHelpers)
    {
        for(int i = 0; i < nHelpers; i++)
            std::thread([this] { Loop(); }).detach();
    }
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
    int mnTasks = 0, mnDone = 0;
    uint32_t mnGeneration = 0;
    std::atomic<uint64_t> mnNext{0};
};

static float busy(int n, unsigned seed)  // ~n * 1 ns of arithmetic with a heap buffer, as the RANSAC loops
{
    std::vector<float> v(256 + seed % 512);
    float s = 0;
    for(int i = 0; i < n; i++)
    {
        v[i % v.size()] += i * 0.5f;
        s += v[(i * 7) % v.size()];
    }
    return s;
}

int main(int argc, char** argv)
{
    const int iters = argc > 1 ? atoi(argv[1]) : 20000, mode = argc > 2 ? atoi(argv[2]) : 0;
    std::ofstream trace(argc > 3 ? argv[3] : "/dev/null");
    std::atomic<bool> stop{false};
    std::atomic<float> sink{0};
    std::thread source([&] {
        while(!stop)
        {
            std::vector<unsigned char> im(752 * 480, 1);
            sink = sink + im[1000];
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    });
    std::thread mapping([&] {
        for(int k = 0; !stop; k++)
        {
            std::vector<std::vector<float>> a(100);
            for(size_t i = 0; i < a.size(); i++)
                a[i].resize(1 + (k * 31 + i * 17) % 2000);
            if(k % 50 == 0)
                std::cout << "LocalMapping " << k << std::endl;
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });
    std::thread closing([&] {
        while(!stop)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });
    WebPool pool(3);
    for(int it = 0; it < iters; it++)
    {
        pool.Run(4, [&](int t) { sink = sink + busy(20000, it * 4 + t); });
        std::cout << "Fail to track local map! " << it << std::endl;
        trace << it << " " << sink.load() << "\n";
        float sh = 0, sf = 0;
        if(mode == 0)
        {
            std::thread th([&] { sh = busy(100000, it); });
            std::thread tf([&] { sf = busy(100000, it + 1); });
            th.join();
            tf.join();
        }
        else
            pool.Run(2, [&](int t) { (t ? sf : sh) = busy(100000, it + t); });
        sink = sink + sh + sf;
    }
    stop = true;
    source.join();
    mapping.join();
    closing.join();
    std::cout << "done " << sink.load() << std::endl;
    return 0;
}
