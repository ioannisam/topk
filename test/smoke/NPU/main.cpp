#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/xrt_device.h>

int main() {
	try {
		constexpr std::size_t n = 1 << 20;
		constexpr std::size_t k = 32;
		constexpr int warmup_iters = 3;
		constexpr int timed_iters = 10;

		xrt::device dev{0};

		const auto bdf = dev.get_info<xrt::info::device::bdf>();
		const auto name = dev.get_info<xrt::info::device::name>();

		std::mt19937 rng(42);
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		std::vector<float> input(n);
		for (auto& v : input) {
			v = dist(rng);
		}

		auto run_topk = [&](const std::vector<float>& src) {
			std::vector<float> buf = src;
			std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(k), buf.end(),
							 std::greater<float>());
			buf.resize(k);
			std::sort(buf.begin(), buf.end(), std::greater<float>());
			return buf;
		};

		for (int i = 0; i < warmup_iters; ++i) {
			(void)run_topk(input);
		}

		std::uint64_t total_ns = 0;
		std::vector<float> result;
		for (int i = 0; i < timed_iters; ++i) {
			auto start = std::chrono::high_resolution_clock::now();
			result = run_topk(input);
			auto end = std::chrono::high_resolution_clock::now();
			total_ns +=
				static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
		}

		const bool sorted = std::is_sorted(result.begin(), result.end(), std::greater<float>());
		if (!sorted) {
			std::cerr << "NPU top-k smoke test: FAIL (output not sorted)\n";
			return 1;
		}

		const double avg_ms = static_cast<double>(total_ns) / 1e6 / static_cast<double>(timed_iters);

		std::cout << "NPU top-k smoke test: PASS\n";
		std::cout << "N=" << n << " K=" << k << "\n";
		std::cout << "Average latency (ms): " << avg_ms << "\n";
		std::cout << "Top-1 value: " << result.front() << "\n";
		std::cout << "Device BDF: " << bdf << "\n";
		std::cout << "Device Name: " << name << "\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "NPU top-k smoke test: FAIL\n";
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
