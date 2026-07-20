#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <chrono>

#include "common/benchmark.hpp"
#include "common/energy.hpp"

namespace common::roofline {

enum class Experiment {
	Stream,
	Sweep,
	Both,
};

struct Config {
	Experiment experiment;
	std::size_t bytes;
	std::vector<int> ops;
	std::size_t ex_threads;
	std::uint64_t seed;
	bool debug_output;
};

struct Point {
	std::string kernel;
	int ops_per_elem;
	std::size_t elements;
	double bytes_moved;
	double flops;
	double ms_mean;
	double ms_stdev;
	double ms_min;
	common::energy::Summary energy;
};

Config parse_args(int argc, char** argv);

template <typename Fn>
Point measure(const char* kernel, int ops, std::size_t n, double bytes_moved, double flops, Fn&& fn) {
	auto best = common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValue<double> {
		asm volatile("" : : : "memory");
		const auto t0 = std::chrono::high_resolution_clock::now();
		common::energy::FullScope energy_scope;

		const double value = fn();
		asm volatile("" : : "r"(&value) : "memory");

		energy_scope.close();
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
		return common::benchmark::TimedValue<double>{elapsed, elapsed, value};
	});

	Point point;
	point.kernel = kernel;
	point.ops_per_elem = ops;
	point.elements = n;
	point.bytes_moved = bytes_moved;
	point.flops = flops;
	point.ms_mean = best.e2e.mean;
	point.ms_stdev = best.e2e.stdev;
	point.ms_min = best.e2e.min;
	point.energy = best.energy;
	return point;
}

double gbytes_per_second(const Point& point);
double gflops_per_second(const Point& point);
double arithmetic_intensity(const Point& point);

void report(const Config& cfg, const char* backend, const std::vector<Point>& points);

} // namespace common::roofline
