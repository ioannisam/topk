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

struct Config {
	int q;
	std::size_t k;
	bool want_max;
	DataType dtype;
	RunMode run_mode;
	bool debug_output;
	std::size_t ex_threads;
	std::uint64_t seed;

	std::string testcase_path;
	bool run_check;
	std::vector<std::string> expected_output_tokens;
	bool has_expected_output;
};

Config parse_args(int argc, char** argv);

} // namespace common::config
