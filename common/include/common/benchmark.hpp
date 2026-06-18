#pragma once

#include <iostream>
#include <utility>

namespace common::benchmark {

constexpr int kWarmupIters = 5;
constexpr int kMeasureIters = 50;

template <typename ValueT> struct TimedValue {
	double elapsed_ms;
	ValueT value;
};

template <typename ValueT, typename StatsT> struct TimedValueWithStats {
	double elapsed_ms;
	ValueT value;
	StatsT stats;
};

template <typename Fn> void warmup(int iterations, Fn&& fn) {
	for (int i = 0; i < iterations; ++i) {
		fn();
	}
}

template <typename Fn> auto measure_best(int iterations, Fn&& fn) {
	const int runs = iterations > 0 ? iterations : 1;
	auto best = fn();
	for (int i = 1; i < runs; ++i) {
		auto current = fn();
		if (current.elapsed_ms < best.elapsed_ms) {
			best = std::move(current);
		}
	}
	return best;
}

template <typename Fn> auto run_benchmark(Fn&& timed_run_once) {
	warmup(kWarmupIters, [&]() { timed_run_once(); });
	auto best = measure_best(kMeasureIters, timed_run_once);
	std::cout << "[PROFILE_TIME_MS] " << best.elapsed_ms << "\n";
	return best;
}

} // namespace common::benchmark
