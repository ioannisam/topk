#include "../include/roofline.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "common/benchmark.hpp"
#include "common/random.hpp"

namespace cpu::roofline {

namespace {

using common::roofline::Config;
using common::roofline::Experiment;
using common::roofline::measure;
using common::roofline::Point;

constexpr std::size_t kLane = 16;

std::size_t worker_count(const Config& cfg, std::size_t n) {
	const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
	std::size_t workers = cfg.ex_threads == 0 ? hw_threads : cfg.ex_threads;
	return std::max<std::size_t>(1, std::min(workers, n));
}

float fma_chunk(const float* src, std::size_t begin, std::size_t end, int ops) {
	float acc[kLane];
	for (std::size_t l = 0; l < kLane; ++l) {
		acc[l] = 0.0f;
	}

	const float a = 1.0000001f;
	const float b = 0.0000001f;

	std::size_t i = begin;
	for (; i + kLane <= end; i += kLane) {
		float v[kLane];
		for (std::size_t l = 0; l < kLane; ++l) {
			v[l] = src[i + l];
		}
		for (int j = 0; j < ops; ++j) {
			for (std::size_t l = 0; l < kLane; ++l) {
				v[l] = v[l] * a + b;
			}
		}
		for (std::size_t l = 0; l < kLane; ++l) {
			acc[l] += v[l];
		}
	}

	float total = 0.0f;
	for (std::size_t l = 0; l < kLane; ++l) {
		total += acc[l];
	}
	for (; i < end; ++i) {
		total += src[i];
	}
	return total;
}

void copy_chunk(const float* src, float* dst, std::size_t begin, std::size_t end) {
	for (std::size_t i = begin; i < end; ++i) {
		dst[i] = src[i];
	}
}

double run_fma(const std::vector<float>& src, std::size_t workers, int ops) {
	const std::size_t n = src.size();
	std::vector<float> partial(workers, 0.0f);
	std::vector<std::thread> pool;
	pool.reserve(workers > 0 ? workers - 1 : 0);

	for (std::size_t tid = 0; tid + 1 < workers; ++tid) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;
			partial[tid] = fma_chunk(src.data(), begin, end, ops);
		});
	}
	const std::size_t last_tid = workers - 1;
	partial[last_tid] = fma_chunk(src.data(), (n * last_tid) / workers, n, ops);

	for (auto& t : pool) {
		t.join();
	}

	double total = 0.0;
	for (const float value : partial) {
		total += static_cast<double>(value);
	}
	return total;
}

double run_copy(const std::vector<float>& src, std::vector<float>& dst, std::size_t workers) {
	const std::size_t n = src.size();
	std::vector<std::thread> pool;
	pool.reserve(workers > 0 ? workers - 1 : 0);

	for (std::size_t tid = 0; tid + 1 < workers; ++tid) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;
			copy_chunk(src.data(), dst.data(), begin, end);
		});
	}
	const std::size_t last_tid = workers - 1;
	copy_chunk(src.data(), dst.data(), (n * last_tid) / workers, n);

	for (auto& t : pool) {
		t.join();
	}

	return static_cast<double>(dst[0]) + static_cast<double>(dst[n - 1]);
}

} // namespace

int execute(const Config& cfg) {
	const std::size_t n = std::max<std::size_t>(kLane, cfg.bytes / sizeof(float));
	const std::size_t workers = worker_count(cfg, n);
	const double read_bytes = static_cast<double>(n) * sizeof(float);

	std::vector<float> src = common::utils::generate_random_input<float>(n, cfg.seed, 0, 1000);
	std::vector<float> dst(n, 0.0f);

	std::vector<Point> points;

	if (cfg.experiment == Experiment::Stream || cfg.experiment == Experiment::Both) {
		points.push_back(
			measure("read", 0, n, read_bytes, static_cast<double>(n), [&]() { return run_fma(src, workers, 0); }));
		points.push_back(measure("copy", 0, n, 2.0 * read_bytes, 0.0, [&]() { return run_copy(src, dst, workers); }));
	}

	if (cfg.experiment == Experiment::Sweep || cfg.experiment == Experiment::Both) {
		for (const int ops : cfg.ops) {
			const double flops = static_cast<double>(n) * (2.0 * static_cast<double>(ops) + 1.0);
			points.push_back(measure("fma", ops, n, read_bytes, flops, [&]() { return run_fma(src, workers, ops); }));
		}
	}

	common::roofline::report(cfg, "cpu", points);
	return 0;
}

} // namespace cpu::roofline
