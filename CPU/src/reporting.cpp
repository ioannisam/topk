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

	common::reporting::print_bitonic_common_metrics(cfg, stats);
	common::reporting::print_key_value("Hardware threads available", hw_threads);
	common::reporting::print_key_value("Execution threads used", ex_threads);
}

void print_map_reduce_debug_metrics(const common::config::Config& cfg, std::size_t hw_threads,
									const common::topk::MapReduceRunStats& stats) {
	if (!cfg.debug_output) {
		return;
	}

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("Hardware threads available", hw_threads);
	common::reporting::print_key_value("Execution threads used", stats.tiles_used);
	common::reporting::print_key_value("Tiles used", stats.tiles_used);
	common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
}

} // namespace cpu::reporting
