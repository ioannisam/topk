#pragma once

#include <cstddef>
#include <cstdint>

enum class DataType {
	Int,
	UInt,
	Float,
	Double,
	Fp16,
};

struct Config {
	int q;
	std::size_t k;
	std::uint64_t seed;
	bool want_max;
	bool debug_output;
	bool run_check;
	std::size_t ex_threads;
	DataType dtype;
};

Config parse_args(int argc, char** argv);
