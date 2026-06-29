#include "common/reporting.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "common/benchmark.hpp"

namespace common::reporting {
namespace {

std::size_t number_width(const std::vector<std::string>& values) {
	std::size_t width = 1;
	for (const std::string& value : values) {
		width = std::max(width, value.size());
	}
	return width;
}

} // namespace

const char* dtype_name(common::config::DataType dtype) {
	switch (dtype) {
	case common::config::DataType::Int:
		return "int";
	case common::config::DataType::UInt:
		return "uint";
	case common::config::DataType::Float:
		return "float";
	case common::config::DataType::Double:
		return "double";
	case common::config::DataType::Fp16:
		return "fp16";
	}
	return "unknown";
}

const char* algorithm_name(common::config::Algorithm algorithm) {
	switch (algorithm) {
	case common::config::Algorithm::Bitonic:
		return "bitonic";
	case common::config::Algorithm::MapReduce:
		return "map_reduce";
	case common::config::Algorithm::GroundTruth:
		return "gt";
	}
	return "unknown";
}

const char* run_mode_name(common::config::RunMode mode) {
	switch (mode) {
	case common::config::RunMode::Trunc:
		return "trunc";
	case common::config::RunMode::Full:
		return "full";
	case common::config::RunMode::Both:
		return "both";
	}
	return "unknown";
}

const char* distribution_name(common::config::Distribution dist) {
	switch (dist) {
	case common::config::Distribution::Uniform:
		return "uniform";
	case common::config::Distribution::Normal:
		return "normal";
	case common::config::Distribution::Sorted:
		return "sorted";
	case common::config::Distribution::Reverse:
		return "reverse";
	}
	return "unknown";
}

std::string format_fixed(double value, int decimals, const char* suffix) {
	std::ostringstream out;
	out << std::fixed << std::setprecision(decimals) << value << suffix;
	return out.str();
}

void print_section_header(const char* title) {
	std::cout << "\n== " << title << " ==\n";
}

void print_key_value(const char* key, const std::string& value) {
	std::cout << "  " << std::left << std::setw(28) << key << ": " << value << "\n";
}

void print_key_value(const char* key, const char* value) {
	print_key_value(key, std::string(value));
}

void print_key_value(const char* key, std::size_t value) {
	std::cout << "  " << std::left << std::setw(28) << key << ": " << value << "\n";
}

void print_key_value(const char* key, double value, int precision) {
	std::ostringstream out;
	out << std::fixed << std::setprecision(precision) << value;
	print_key_value(key, out.str());
}

void print_configuration(const common::config::Config& cfg, std::size_t n, std::optional<std::size_t> ex_threads,
						 const char* run_mode_label, const char* backend_tag) {
	print_section_header("Configuration");
	if (backend_tag != nullptr && backend_tag[0] != '\0') {
		print_key_value("Backend", backend_tag);
	}
	print_key_value("Input size N (2^q)", n);
	if (ex_threads.has_value()) {
		print_key_value("Execution threads", ex_threads.value());
	}
	print_key_value("Requested top-k", cfg.k);
	print_key_value("Data type", dtype_name(cfg.dtype));
	print_key_value("Algorithm", algorithm_name(cfg.algorithm));
	print_key_value("Mode", (cfg.want_max ? "max" : "min"));
	print_key_value(run_mode_label, run_mode_name(cfg.run_mode));
	print_key_value("Debug output", (cfg.debug_output ? "on" : "off"));
	print_key_value("CPU reference verify", (cfg.verify_output ? "on" : "off"));
	print_key_value("Random range", std::to_string(cfg.rand_min) + ".." + std::to_string(cfg.rand_max));
	print_key_value("Input distribution", distribution_name(cfg.dist));
	print_key_value("Benchmark iterations",
					static_cast<std::size_t>(common::benchmark::kWarmupIters + common::benchmark::kMeasureIters));
}

void print_timing_lines(const std::vector<std::pair<std::string, std::optional<double>>>& lines) {
	print_section_header("Timing");
	for (const auto& [label, value] : lines) {
		if (value.has_value()) {
			print_key_value(label.c_str(), value.value(), 3);
		} else {
			print_key_value(label.c_str(), "skipped");
		}
	}
}

void print_bitonic_common_metrics(const common::config::Config& cfg, const common::topk::BitonicRunStats& stats) {
	const std::size_t full_cmp = stats.full_comparators;
	const std::size_t trunc_cmp = stats.trunc_comparators;
	const double full_algo_ms = stats.full_run_stats != nullptr ? stats.full_run_stats->algorithm_ms : 0.0;
	const double trunc_algo_ms = stats.trunc_run_stats != nullptr ? stats.trunc_run_stats->algorithm_ms : 0.0;

	const std::size_t skipped = full_cmp >= trunc_cmp ? (full_cmp - trunc_cmp) : 0;
	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(skipped) / static_cast<double>(full_cmp));
	const bool ran_both = cfg.run_mode == common::config::RunMode::Both;
	const double speedup = (ran_both && trunc_algo_ms > 0.0) ? (full_algo_ms / trunc_algo_ms) : 0.0;

	print_section_header("Debug Metrics");
	print_key_value("Bitonic layers", stats.layer_count);
	print_key_value("Full comparators", full_cmp);
	print_key_value("Trunc comparators", trunc_cmp);
	print_key_value("Comparator skip ratio", format_fixed(skipped_pct, 2, "%"));
	if (ran_both) {
		print_key_value("Full/Trunc speedup", format_fixed(speedup, 3, "x"));
		print_key_value("Speedup interpretation", (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

void print_check_result(const char* label, bool enabled, bool ok) {
	if (!enabled) {
		return;
	}
	std::cout << label << ": " << (ok ? "OK" : "FAIL") << "\n";
}

void print_output(const common::config::Config& cfg, const std::vector<std::string>& output) {
	print_section_header("Top-k Output");
	print_key_value("Mode", (cfg.want_max ? "max" : "min"));
	print_key_value("Order", "sorted");
	print_key_value("Values reported", output.size());

	if (output.empty()) {
		std::cout << "  Values: (none)\n";
		return;
	}

	constexpr std::size_t values_per_row = 16;
	const std::size_t width = number_width(output);

	// With debug off, only preview the first row so large-k runs do not dump
	// thousands of values into stdout (and into the profiler's captured output).
	const std::size_t shown = cfg.debug_output ? output.size() : std::min(output.size(), values_per_row);

	std::cout << "  Values:\n";
	for (std::size_t i = 0; i < shown; ++i) {
		if (i % values_per_row == 0) {
			std::cout << "    ";
		}
		std::cout << std::right << std::setw(static_cast<int>(width)) << output[i];
		const bool end_of_row = (i + 1) % values_per_row == 0;
		const bool last_value = (i + 1) == shown;
		if (end_of_row || last_value) {
			std::cout << "\n";
		} else {
			std::cout << " ";
		}
	}
	if (shown < output.size()) {
		std::cout << "    ... (" << (output.size() - shown) << " more; enable debug=true for the full list)\n";
	}

	std::cout << std::left;
}

} // namespace common::reporting
