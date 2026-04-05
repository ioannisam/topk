#include "reporting.hpp"

#include <sstream>
#include <string>

#include "common/reporting.hpp"

void print_configuration(const common::Config& cfg, std::size_t ex_threads, std::size_t n) {
	common::reporting::print_configuration(cfg, n, ex_threads, "Network run mode", "cpu");
}

void print_timing_summary(bool ran_full, bool ran_trunc, double full_ms, double trunc_ms) {
	common::reporting::print_timing_lines({
		{"Full bitonic time (ms)", ran_full ? std::optional<double>(full_ms) : std::nullopt},
		{"Trunc bitonic time (ms)", ran_trunc ? std::optional<double>(trunc_ms) : std::nullopt},
	});
}

void print_skipped_summary(bool ran_trunc, std::size_t full_cmp, std::size_t trunc_cmp) {
	if (!ran_trunc) {
		return;
	}

	const std::size_t skipped = full_cmp >= trunc_cmp ? (full_cmp - trunc_cmp) : 0;
	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(skipped) / static_cast<double>(full_cmp));
	std::ostringstream value;
	value << skipped << "/" << full_cmp << " (" << common::reporting::format_fixed(skipped_pct, 2, "%") << ")";
	common::reporting::print_key_value("Skipped comparators", value.str());
}

void print_debug_metrics(const common::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
						 std::size_t layer_count,
						 std::size_t full_cmp, std::size_t trunc_cmp, double full_ms, double trunc_ms) {
	if (!cfg.debug_output) {
		return;
	}

	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));
	const bool ran_both = cfg.run_mode == common::RunMode::Both;
	const double speedup = (ran_both && trunc_ms > 0.0) ? (full_ms / trunc_ms) : 0.0;

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("Hardware threads available", hw_threads);
	common::reporting::print_key_value("Execution threads used", ex_threads);
	common::reporting::print_key_value("Bitonic layers", layer_count);
	common::reporting::print_key_value("Full comparators", full_cmp);
	common::reporting::print_key_value("Trunc comparators", trunc_cmp);
	common::reporting::print_key_value("Comparator skip ratio", common::reporting::format_fixed(skipped_pct, 2, "%"));
	if (ran_both) {
		common::reporting::print_key_value("Full/Trunc speedup", common::reporting::format_fixed(speedup, 3, "x"));
		common::reporting::print_key_value("Speedup interpretation", (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

void print_correctness_summary(bool run_check, bool ok) {
	common::reporting::print_check_result("Top-k correctness vs full network", run_check, ok);
}

void print_output(const common::Config& cfg, const std::vector<std::string>& output) {
	common::reporting::print_output(cfg, output);
}
