#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

double gbytes_per_second(const Point& point);
double gflops_per_second(const Point& point);
double arithmetic_intensity(const Point& point);

void report(const Config& cfg, const char* backend, const std::vector<Point>& points);

} // namespace common::roofline
