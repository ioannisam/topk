#include "common/roofline.hpp"

#include "parse_utils.hpp"

#include <cctype>
#include <cstdio>
#include <stdexcept>

#include "common/reporting.hpp"

#ifndef DEBUG
#define DEBUG 0
#endif

namespace common::roofline {
namespace {

constexpr const char* kUsage = "Usage: ./roofline [exp=stream|sweep|cache|transfer|both|all] [bytes=<size>[K|M|G]] "
							   "[ops=<csv>] [sizes=<csv of sizes>] [threads=<num>] [seed=<seed>] [debug=true|false]";

using common::parse::parse_bool_value;
using common::parse::parse_signed_long;

std::size_t parse_bytes(const std::string& text) {
	if (text.empty()) {
		throw std::invalid_argument("bytes must not be empty");
	}

	std::size_t multiplier = 1;
	std::string digits = text;
	const char suffix = static_cast<char>(std::toupper(static_cast<unsigned char>(text.back())));
	if (suffix == 'K' || suffix == 'M' || suffix == 'G') {
		multiplier =
			suffix == 'K' ? (std::size_t{1} << 10) : (suffix == 'M' ? (std::size_t{1} << 20) : (std::size_t{1} << 30));
		digits = text.substr(0, text.size() - 1);
	}

	const long long value = parse_signed_long(digits, "bytes");
	if (value <= 0) {
		throw std::invalid_argument("bytes must be positive");
	}
	return static_cast<std::size_t>(value) * multiplier;
}

std::vector<int> parse_ops(const std::string& text) {
	std::vector<int> out;
	std::string current;
	for (const char c : text) {
		if (c == ',') {
			if (!current.empty()) {
				out.push_back(static_cast<int>(parse_signed_long(current, "ops")));
				current.clear();
			}
		} else {
			current.push_back(c);
		}
	}
	if (!current.empty()) {
		out.push_back(static_cast<int>(parse_signed_long(current, "ops")));
	}

	for (const int value : out) {
		if (value < 0) {
			throw std::invalid_argument("ops must be non-negative");
		}
	}
	if (out.empty()) {
		throw std::invalid_argument("ops must list at least one value");
	}
	return out;
}

std::vector<std::size_t> parse_sizes(const std::string& text) {
	std::vector<std::size_t> out;
	std::string current;
	for (const char c : text) {
		if (c == ',') {
			if (!current.empty()) {
				out.push_back(parse_bytes(current));
				current.clear();
			}
		} else {
			current.push_back(c);
		}
	}
	if (!current.empty()) {
		out.push_back(parse_bytes(current));
	}
	if (out.empty()) {
		throw std::invalid_argument("sizes must list at least one value");
	}
	return out;
}

std::vector<std::size_t> default_sizes(std::size_t bytes) {
	std::vector<std::size_t> out;
	for (std::size_t size = std::size_t{32} << 10; size < bytes; size *= 2) {
		out.push_back(size);
	}
	out.push_back(bytes);
	return out;
}

Experiment parse_experiment(const std::string& value) {
	if (value == "stream") {
		return Experiment::Stream;
	}
	if (value == "sweep") {
		return Experiment::Sweep;
	}
	if (value == "cache") {
		return Experiment::Cache;
	}
	if (value == "transfer") {
		return Experiment::Transfer;
	}
	if (value == "both") {
		return Experiment::Both;
	}
	if (value == "all") {
		return Experiment::All;
	}
	throw std::invalid_argument("Unsupported experiment. Use one of: stream, sweep, cache, transfer, both, all");
}

} // namespace

Config parse_args(int argc, char** argv) {
	Config cfg{};
	cfg.experiment = Experiment::Both;
	cfg.bytes = std::size_t{512} << 20;
	cfg.ops = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256};
	cfg.ex_threads = 0;
	cfg.seed = 42;
	cfg.debug_output = DEBUG != 0;

	for (int i = 1; i < argc; ++i) {
		const std::string token = argv[i];
		if (token.find('=') == std::string::npos) {
			throw std::invalid_argument(kUsage);
		}

		if (token.starts_with("exp=")) {
			cfg.experiment = parse_experiment(token.substr(4));
			continue;
		}
		if (token.starts_with("bytes=")) {
			cfg.bytes = parse_bytes(token.substr(6));
			continue;
		}
		if (token.starts_with("ops=")) {
			cfg.ops = parse_ops(token.substr(4));
			continue;
		}
		if (token.starts_with("sizes=")) {
			cfg.sizes = parse_sizes(token.substr(6));
			continue;
		}
		if (token.starts_with("threads=")) {
			const long long parsed = parse_signed_long(token.substr(8), "threads");
			if (parsed <= 0) {
				throw std::invalid_argument("threads must be positive");
			}
			cfg.ex_threads = static_cast<std::size_t>(parsed);
			continue;
		}
		if (token.starts_with("seed=")) {
			const long long parsed = parse_signed_long(token.substr(5), "seed");
			if (parsed < 0) {
				throw std::invalid_argument("seed must be non-negative");
			}
			cfg.seed = static_cast<std::uint64_t>(parsed);
			continue;
		}
		if (token.starts_with("debug=")) {
			cfg.debug_output = parse_bool_value(token.substr(6), "debug");
			continue;
		}

		throw std::invalid_argument("Unknown key token: " + token);
	}

	if (cfg.sizes.empty()) {
		cfg.sizes = default_sizes(cfg.bytes);
	}

	return cfg;
}

double gbytes_per_second(const Point& point) {
	if (point.ms_min <= 0.0) {
		return 0.0;
	}
	return point.bytes_moved / (point.ms_min * 1e6);
}

double gops_per_second(const Point& point) {
	if (point.ms_min <= 0.0) {
		return 0.0;
	}
	return point.ops / (point.ms_min * 1e6);
}

double operational_intensity(const Point& point) {
	if (point.bytes_moved <= 0.0) {
		return 0.0;
	}
	return point.ops / point.bytes_moved;
}

void report(const Config& cfg, const char* backend, const std::vector<Point>& points) {
	if (points.empty()) {
		std::fprintf(stderr, "warning: backend %s produced no roofline points for the selected experiment\n", backend);
	}
	if (cfg.debug_output) {
		common::reporting::print_section_header("Roofline Configuration");
		common::reporting::print_key_value("Backend", backend);
		common::reporting::print_key_value("Working set (MiB)", static_cast<double>(cfg.bytes) / (1024.0 * 1024.0), 1);
		common::reporting::print_key_value("Threads", cfg.ex_threads);
		common::reporting::print_section_header("Roofline Points");
	}

	std::printf("ROOFLINE_SCHEMA,backend,kernel,ops_per_elem,elements,bytes_moved,ops,ms_mean,ms_stdev,ms_min,"
				"gbytes_per_s,gops_per_s,operational_intensity,joules_per_iter\n");

	for (const Point& point : points) {
		const int iterations = point.energy.iterations > 0 ? point.energy.iterations : 1;
		const double joules =
			point.energy.available ? point.energy.algo_total.total() / static_cast<double>(iterations) : 0.0;

		std::printf("ROOFLINE,%s,%s,%d,%zu,%.0f,%.0f,%.6f,%.6f,%.6f,%.4f,%.4f,%.6f,%.6f\n", backend,
					point.kernel.c_str(), point.ops_per_elem, point.elements, point.bytes_moved, point.ops,
					point.ms_mean, point.ms_stdev, point.ms_min, gbytes_per_second(point), gops_per_second(point),
					operational_intensity(point), joules);
	}

	if (cfg.debug_output) {
		for (const Point& point : points) {
			const std::string label = point.kernel + " (ops/elem=" + std::to_string(point.ops_per_elem) + ")";
			common::reporting::print_key_value(label.c_str(),
											   common::reporting::format_fixed(gbytes_per_second(point), 2, " GB/s"));
		}
	}
}

} // namespace common::roofline
