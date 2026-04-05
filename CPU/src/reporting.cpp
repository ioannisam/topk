#include "reporting.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void print_section_header(const char* title) {
	std::cout << "\n== " << title << " ==\n";
}

template <typename T> void print_key_value(const char* key, const T& value) {
	std::cout << "  " << std::left << std::setw(28) << key << ": " << value << "\n";
}

std::string format_fixed(double value, int decimals, const char* suffix = "") {
	std::ostringstream out;
	out << std::fixed << std::setprecision(decimals) << value << suffix;
	return out.str();
}

std::size_t number_width(const std::vector<std::string>& values) {
	std::size_t width = 1;
	for (const std::string& value : values) {
		width = std::max(width, value.size());
	}
	return width;
}

const char* dtype_name(DataType dtype) {
	switch (dtype) {
	case DataType::Int:
		return "int";
	case DataType::UInt:
		return "uint";
	case DataType::Float:
		return "float";
	case DataType::Double:
		return "double";
	case DataType::Fp16:
		return "fp16";
	}
	return "unknown";
}

const char* run_mode_name(RunMode mode) {
	switch (mode) {
	case RunMode::Trunc:
		return "trunc";
	case RunMode::Full:
		return "full";
	case RunMode::Both:
		return "both";
	}
	return "unknown";
}

} // namespace

void print_configuration(const Config& cfg, std::size_t ex_threads, std::size_t n) {
	print_section_header("Configuration");
	print_key_value("Input size N (2^q)", n);
	print_key_value("Execution threads", ex_threads);
	print_key_value("Requested top-k", cfg.k);
	print_key_value("Data type", dtype_name(cfg.dtype));
	print_key_value("Mode", (cfg.want_max ? "max" : "min"));
	print_key_value("Network run mode", run_mode_name(cfg.run_mode));
	print_key_value("Debug output", (cfg.debug_output ? "on" : "off"));
	if (!cfg.testcase_path.empty()) {
		print_key_value("Testcase answer check", (cfg.run_check ? "on" : "off"));
	}
	if (!cfg.testcase_path.empty()) {
		print_key_value("Testcase file", std::filesystem::path(cfg.testcase_path).filename().string());
	}
}

void print_timing_summary(bool ran_full, bool ran_trunc, double full_ms, double trunc_ms) {
	print_section_header("Timing");
	std::cout << std::fixed << std::setprecision(3);
	if (ran_full) {
		print_key_value("Full bitonic time (ms)", full_ms);
	} else {
		print_key_value("Full bitonic time (ms)", "skipped");
	}
	if (ran_trunc) {
		print_key_value("Trunc bitonic time (ms)", trunc_ms);
	} else {
		print_key_value("Trunc bitonic time (ms)", "skipped");
	}
	std::cout << std::defaultfloat;
}

void print_skipped_summary(bool ran_trunc, std::size_t full_cmp, std::size_t trunc_cmp) {
	if (!ran_trunc) {
		return;
	}

	const std::size_t skipped = full_cmp >= trunc_cmp ? (full_cmp - trunc_cmp) : 0;
	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(skipped) / static_cast<double>(full_cmp));
	std::ostringstream value;
	value << skipped << "/" << full_cmp << " (" << format_fixed(skipped_pct, 2, "%") << ")";
	print_key_value("Skipped comparators", value.str());
}

void print_debug_metrics(const Config& cfg, std::size_t hw_threads, std::size_t ex_threads, std::size_t layer_count,
						 std::size_t full_cmp, std::size_t trunc_cmp, double full_ms, double trunc_ms) {
	if (!cfg.debug_output) {
		return;
	}

	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));
	const bool ran_both = cfg.run_mode == RunMode::Both;
	const double speedup = (ran_both && trunc_ms > 0.0) ? (full_ms / trunc_ms) : 0.0;

	print_section_header("Debug Metrics");
	print_key_value("Hardware threads available", hw_threads);
	print_key_value("Execution threads used", ex_threads);
	print_key_value("Bitonic layers", layer_count);
	print_key_value("Full comparators", full_cmp);
	print_key_value("Trunc comparators", trunc_cmp);
	print_key_value("Comparator skip ratio", format_fixed(skipped_pct, 2, "%"));
	if (ran_both) {
		print_key_value("Full/Trunc speedup", format_fixed(speedup, 3, "x"));
		print_key_value("Speedup interpretation", (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

void print_correctness_summary(bool run_check, bool ok) {
	if (!run_check) {
		return;
	}
	std::cout << "Top-k correctness vs full network: " << (ok ? "OK" : "FAIL") << "\n";
}

void print_topk_output(const Config& cfg, const std::vector<std::string>& output) {
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

	std::cout << "  Values:\n";
	for (std::size_t i = 0; i < output.size(); ++i) {
		if (i % values_per_row == 0) {
			std::cout << "    ";
		}
		std::cout << std::right << std::setw(static_cast<int>(width)) << output[i];
		const bool end_of_row = (i + 1) % values_per_row == 0;
		const bool last_value = (i + 1) == output.size();
		if (end_of_row || last_value) {
			std::cout << "\n";
		} else {
			std::cout << " ";
		}
	}

	std::cout << std::left;
}
