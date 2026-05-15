#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace common::config {

enum class DataType {
	Int,
	UInt,
	Float,
	Double,
	Fp16,
};

enum class RunMode {
	Trunc,
	Full,
	Both,
};

enum class Algorithm {
	Bitonic,
	MapReduce,
    GroundTruth,
};

struct Config {
	int q;
	std::size_t k;
	bool want_max;
	DataType dtype;
	Algorithm algorithm;
	RunMode run_mode;
	bool debug_output;
	std::size_t ex_threads;
	std::uint64_t seed;
	bool verify_output;
	int rand_min;
	int rand_max;
};

Config parse_args(int argc, char** argv);

} // namespace common::config
