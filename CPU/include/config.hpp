#pragma once

#include <cstddef>
#include <cstdint>

struct Config {
	int q;
	std::size_t k;
	std::uint64_t seed;
	bool want_max;
	bool sort_output;
	bool debug_output;
	bool run_check;
	std::size_t ex_threads;
};

Config parse_args(int argc, char** argv);
