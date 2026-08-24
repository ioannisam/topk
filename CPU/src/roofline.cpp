#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

#include "common/benchmark.hpp"
#include "common/random.hpp"
#include "common/roofline.hpp"

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
	for (std::size_t l = 0; l < kLane; l++) {
		acc[l] = 0.0f;
	}

	const float a = 1.0000001f;
	const float b = 0.0000001f;

	std::size_t i = begin;
	for (; i + kLane <= end; i += kLane) {
		float v[kLane];
		for (std::size_t l = 0; l < kLane; l++) {
			v[l] = src[i + l];
		}
		for (int j = 0; j < ops; j++) {
			for (std::size_t l = 0; l < kLane; l++) {
				v[l] = v[l] * a + b;
			}
		}
		for (std::size_t l = 0; l < kLane; l++) {
			acc[l] += v[l];
		}
	}

	float total = 0.0f;
	for (std::size_t l = 0; l < kLane; l++) {
		total += acc[l];
	}
	for (; i < end; i++) {
		total += src[i];
	}
	return total;
}

constexpr std::size_t kCmpLane = 64;
float cmp_chunk(const float* src, std::size_t begin, std::size_t end, int ops) {
	float x[kCmpLane];
	float y[kCmpLane];
	float z[kCmpLane];
	float acc[kCmpLane];
	for (std::size_t l = 0; l < kCmpLane; l++) {
		acc[l] = 0.0f;
		z[l] = static_cast<float>(l);
	}

	std::size_t i = begin;
	for (; i + 2 * kCmpLane <= end; i += 2 * kCmpLane) {
		for (std::size_t l = 0; l < kCmpLane; l++) {
			x[l] = src[i + l];
			y[l] = src[i + kCmpLane + l];
		}
		// Three-register rotation keeps every step a real compare-exchange the compiler cannot fold away.
		for (int j = 0; j < ops; j++) {
			for (std::size_t l = 0; l < kCmpLane; l++) {
				const float lo = std::min(x[l], y[l]);
				const float hi = std::max(x[l], y[l]);
				x[l] = lo;
				y[l] = z[l];
				z[l] = hi;
			}
		}
		for (std::size_t l = 0; l < kCmpLane; l++) {
			acc[l] += x[l] + y[l];
		}
	}

	float total = 0.0f;
	for (std::size_t l = 0; l < kCmpLane; l++) {
		total += acc[l] + z[l];
	}
	for (; i < end; i++) {
		total += src[i];
	}
	return total;
}

void copy_chunk(const float* src, float* dst, std::size_t begin, std::size_t end) {
	for (std::size_t i = begin; i < end; i++) {
		dst[i] = src[i];
	}
}

double run_fma(const float* src, std::size_t n, std::size_t workers, int ops, std::size_t repeats = 1) {
	std::vector<float> partial(workers, 0.0f);
	std::vector<std::thread> pool;
	pool.reserve(workers > 0 ? workers - 1 : 0);

	for (std::size_t tid = 0; tid + 1 < workers; tid++) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;
			float acc = 0.0f;
			for (std::size_t r = 0; r < repeats; r++) {
				acc += fma_chunk(src, begin, end, ops);
				asm volatile("" : : : "memory");
			}
			partial[tid] = acc;
		});
	}
	const std::size_t last_tid = workers - 1;
	const std::size_t last_begin = (n * last_tid) / workers;
	float last_acc = 0.0f;
	for (std::size_t r = 0; r < repeats; r++) {
		last_acc += fma_chunk(src, last_begin, n, ops);
		asm volatile("" : : : "memory");
	}
	partial[last_tid] = last_acc;

	for (auto& t : pool) {
		t.join();
	}

	double total = 0.0;
	for (const float value : partial) {
		total += static_cast<double>(value);
	}
	return total;
}

double run_cmp(const float* src, std::size_t n, std::size_t workers, int ops) {
	std::vector<float> partial(workers, 0.0f);
	std::vector<std::thread> pool;
	pool.reserve(workers > 0 ? workers - 1 : 0);

	for (std::size_t tid = 0; tid + 1 < workers; tid++) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;
			partial[tid] += cmp_chunk(src, begin, end, ops);
		});
	}
	const std::size_t last_tid = workers - 1;
	partial[last_tid] += cmp_chunk(src, (n * last_tid) / workers, n, ops);

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

	for (std::size_t tid = 0; tid + 1 < workers; tid++) {
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

	if (common::roofline::includes(cfg.experiment, Experiment::Stream)) {
		points.push_back(measure("read", 0, n, read_bytes, static_cast<double>(n), [&]() {
			return run_fma(src.data(), n, workers, 0);
		}));
		points.push_back(measure("copy", 0, n, 2.0 * read_bytes, 0.0, [&]() { return run_copy(src, dst, workers); }));
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Sweep)) {
		for (const int ops : cfg.ops) {
			const double flops = static_cast<double>(n) * (2.0 * static_cast<double>(ops) + 1.0);
			points.push_back(measure("fma", ops, n, read_bytes, flops, [&]() {
				return run_fma(src.data(), n, workers, ops);
			}));
			const double cmp_ops = common::roofline::compare_exchange_count(n, ops);
			points.push_back(measure("cmp", ops, n, read_bytes, cmp_ops, [&]() {
				return run_cmp(src.data(), n, workers, ops);
			}));
		}
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Cache)) {
		for (const std::size_t size : cfg.sizes) {
			const std::size_t elems = std::max<std::size_t>(kLane, size / sizeof(float));
			if (elems > n) {
				continue;
			}
			const std::size_t threads = worker_count(cfg, elems);
			const std::size_t repeats = std::max<std::size_t>(1, n / elems);
			const double moved = static_cast<double>(elems) * sizeof(float) * static_cast<double>(repeats);
			points.push_back(measure("cache_read", 0, elems, moved, 0.0, [&]() {
				return run_fma(src.data(), elems, threads, 0, repeats);
			}));
		}
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Transfer)) {
		for (const std::size_t size : cfg.sizes) {
			const std::size_t elems = std::max<std::size_t>(1, size / sizeof(float));
			if (elems > n) {
				continue;
			}
			const double moved = static_cast<double>(elems) * sizeof(float);
			points.push_back(measure("stage_write", 0, elems, moved, 0.0, [&]() {
				std::memcpy(dst.data(), src.data(), elems * sizeof(float));
				return 0.0;
			}));
		}
	}

	common::roofline::report(cfg, "cpu", points);
	return 0;
}

} // namespace cpu::roofline

int main(int argc, char** argv) {
	try {
		const common::roofline::Config cfg = common::roofline::parse_args(argc, argv);
		return cpu::roofline::execute(cfg);
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
