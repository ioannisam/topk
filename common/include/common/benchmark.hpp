#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/energy.hpp"

namespace common::benchmark {

constexpr int kWarmupIters = 5;
constexpr int kMeasureIters = 50;

template <typename ValueT> struct TimedValue {
	double e2e_ms;
	double algo_ms;
	ValueT value;
};

template <typename ValueT, typename StatsT> struct TimedValueWithStats {
	double e2e_ms;
	double algo_ms;
	ValueT value;
	StatsT stats;
};

struct ChannelSummary {
	double mean = 0.0;
	double stdev = 0.0;
	double min = 0.0;
};

inline ChannelSummary summarize(std::vector<double> samples) {
	ChannelSummary summary;
	if (samples.empty()) {
		return summary;
	}

	std::sort(samples.begin(), samples.end());
	const std::size_t n = samples.size();

	double sum = 0.0;
	for (const double x : samples) {
		sum += x;
	}
	summary.mean = sum / static_cast<double>(n);

	double acc = 0.0;
	for (const double x : samples) {
		const double d = x - summary.mean;
		acc += d * d;
	}
	summary.stdev = std::sqrt(acc / static_cast<double>(n));

	summary.min = samples.front();

	return summary;
}

template <typename SampleT> struct BenchmarkResult {
	ChannelSummary e2e;
	ChannelSummary algo;
	common::energy::Summary energy;
	SampleT sample;
};

template <typename Fn> void warmup(int iterations, Fn&& fn) {
	for (int i = 0; i < iterations; i++) {
		fn();
	}
}

template <typename Fn> auto run_benchmark(Fn&& timed_run_once) {
	using SampleT = std::decay_t<decltype(timed_run_once())>;

	warmup(kWarmupIters, [&]() { timed_run_once(); });

	const int runs = kMeasureIters > 0 ? kMeasureIters : 1;
	std::vector<double> e2e_samples;
	std::vector<double> algo_samples;
	e2e_samples.reserve(static_cast<std::size_t>(runs));
	algo_samples.reserve(static_cast<std::size_t>(runs));

	common::energy::reset_accumulators();
	auto& energy_counter = common::energy::counter();
	const common::energy::Sample loop_start = energy_counter.read();
	const auto loop_t0 = std::chrono::steady_clock::now();

	SampleT representative = timed_run_once();
	e2e_samples.push_back(representative.e2e_ms);
	algo_samples.push_back(representative.algo_ms);

	for (int i = 1; i < runs; i++) {
		SampleT current = timed_run_once();
		e2e_samples.push_back(current.e2e_ms);
		algo_samples.push_back(current.algo_ms);
		if (current.e2e_ms < representative.e2e_ms) {
			representative = std::move(current);
		}
	}

	const auto loop_t1 = std::chrono::steady_clock::now();

	BenchmarkResult<SampleT> result;
	result.energy.loop_total = energy_counter.read() - loop_start;
	result.energy.loop_seconds = std::chrono::duration<double>(loop_t1 - loop_t0).count();
	result.energy.e2e_total = common::energy::take_accumulator(common::energy::Channel::E2e);
	result.energy.algo_total = common::energy::take_accumulator(common::energy::Channel::Algo);
	result.energy.e2e_seconds = common::energy::take_seconds(common::energy::Channel::E2e);
	result.energy.algo_seconds = common::energy::take_seconds(common::energy::Channel::Algo);
	result.energy.wait_total = common::energy::take_accumulator(common::energy::Channel::Wait);
	result.energy.wait_seconds = common::energy::take_seconds(common::energy::Channel::Wait);
	result.energy.wait_count = common::energy::take_count(common::energy::Channel::Wait);
	result.energy.iterations = runs;
	result.energy.available = energy_counter.available();
	result.e2e = summarize(std::move(e2e_samples));
	result.algo = summarize(std::move(algo_samples));
	result.sample = std::move(representative);
	return result;
}

} // namespace common::benchmark
