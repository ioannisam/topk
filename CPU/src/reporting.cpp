#include "../include/reporting.hpp"

#include "common/reporting.hpp"

namespace cpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t ex_threads, std::size_t n) {
	common::reporting::print_configuration(cfg, n, ex_threads, "Network run mode", "cpu");
}

void print_bitonic_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
								 const common::topk::BitonicRunStats& stats) {
	if (!cfg.debug_output) {
		return;
	}

	const std::size_t full_cmp = stats.full_comparators;
	const std::size_t trunc_cmp = stats.trunc_comparators;
	const double full_algo_ms = stats.full_run_stats != nullptr ? stats.full_run_stats->algorithm_ms : 0.0;
	const double trunc_algo_ms = stats.trunc_run_stats != nullptr ? stats.trunc_run_stats->algorithm_ms : 0.0;

	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));
	const bool ran_both = cfg.run_mode == common::config::RunMode::Both;
	const double speedup = (ran_both && trunc_algo_ms > 0.0) ? (full_algo_ms / trunc_algo_ms) : 0.0;

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("Hardware threads available", hw_threads);
	common::reporting::print_key_value("Execution threads used", ex_threads);
	common::reporting::print_key_value("Bitonic layers", stats.layer_count);
	common::reporting::print_key_value("Full comparators", full_cmp);
	common::reporting::print_key_value("Trunc comparators", trunc_cmp);
	common::reporting::print_key_value("Comparator skip ratio", common::reporting::format_fixed(skipped_pct, 2, "%"));
	if (ran_both) {
		common::reporting::print_key_value("Full/Trunc speedup", common::reporting::format_fixed(speedup, 3, "x"));
		common::reporting::print_key_value("Speedup interpretation",
										   (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

void print_map_reduce_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads, std::size_t ex_threads,
									const common::topk::MapReduceRunStats& stats) {
	if (!cfg.debug_output) {
		return;
	}

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("Hardware threads available", hw_threads);
	common::reporting::print_key_value("Execution threads used", ex_threads);
	common::reporting::print_key_value("Tiles used", stats.tiles_used);
	common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
}

} // namespace cpu::reporting
