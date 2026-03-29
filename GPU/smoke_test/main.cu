#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

int main() {
    constexpr std::size_t n = 1 << 20;
    constexpr std::size_t k = 32;
    constexpr int warmup_iters = 3;
    constexpr int timed_iters = 10;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> host_input(n);
    for (auto& v : host_input) {
        v = dist(rng);
    }

    thrust::device_vector<float> dvec(host_input.begin(), host_input.end());
    thrust::host_vector<float> topk(k);

    auto run_topk = [&]() {
        thrust::device_vector<float> work = dvec;
        thrust::sort(work.begin(), work.end(), thrust::greater<float>());
        thrust::copy(work.begin(), work.begin() + static_cast<std::ptrdiff_t>(k), topk.begin());
    };

    for (int i = 0; i < warmup_iters; ++i) {
        run_topk();
    }

    std::uint64_t total_ns = 0;
    for (int i = 0; i < timed_iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        run_topk();
        auto end = std::chrono::high_resolution_clock::now();
        total_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    const bool sorted = std::is_sorted(topk.begin(), topk.end(), std::greater<float>());
    if (!sorted) {
        std::cerr << "GPU top-k smoke test: FAIL (output not sorted)\n";
        return 1;
    }

    const double avg_ms = static_cast<double>(total_ns) / 1e6 / static_cast<double>(timed_iters);
    std::cout << "GPU top-k smoke test: PASS\n";
    std::cout << "N=" << n << " K=" << k << "\n";
    std::cout << "Average latency (ms): " << avg_ms << "\n";
    std::cout << "Top-1 value: " << topk.front() << "\n";
    return 0;
}
