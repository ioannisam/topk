#include "../include/reporting.hpp"

#include "common/reporting.hpp"

namespace cpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t ex_threads, std::size_t n) {
	common::reporting::print_configuration(cfg, n, ex_threads, "Network run mode", "cpu");
}

void print_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
						 std::size_t layer_count,
						 std::size_t full_cmp, std::size_t trunc_cmp, double full_ms, double trunc_ms) {
	if (!cfg.debug_output) {
		return;
	}

	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));
	const bool ran_both = cfg.run_mode == common::config::RunMode::Both;
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

} // namespace cpu::reporting
