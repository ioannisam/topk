#include "../include/reporting.hpp"

#include "common/reporting.hpp"

namespace npu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, std::size_t ex_threads,
						 const std::string& device_name, const std::string& device_bdf, bool offload_enabled) {
	common::reporting::print_configuration(cfg, n, ex_threads, "Network run mode", "npu");
	common::reporting::print_key_value("NPU device", device_name);
	common::reporting::print_key_value("NPU BDF", device_bdf);
	common::reporting::print_key_value("Offload configured", (offload_enabled ? "yes" : "no"));
}

void print_bitonic_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
                                 const common::topk::BitonicRunStats& stats,
						         const npu::bitonic::RunStats* full_stats, 
                                 const npu::bitonic::RunStats* trunc_stats) {
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
	common::reporting::print_key_value("NPU device", device_name);
	common::reporting::print_key_value("Bitonic layers", stats.layer_count);
	common::reporting::print_key_value("Full comparators", full_cmp);
	common::reporting::print_key_value("Trunc comparators", trunc_cmp);
	common::reporting::print_key_value("Comparator skip ratio", common::reporting::format_fixed(skipped_pct, 2, "%"));

	if (full_stats != nullptr) {
		common::reporting::print_key_value("Full NPU dispatches", full_stats->layer_dispatches);
		common::reporting::print_key_value("Full execution path",
										   (full_stats->used_offload ? "xrt offload" : "not executed"));
		common::reporting::print_key_value("Execution workers", full_stats->workers);
	}
	if (trunc_stats != nullptr) {
		common::reporting::print_key_value("Trunc NPU dispatches", trunc_stats->layer_dispatches);
		common::reporting::print_key_value("Trunc execution path",
										   (trunc_stats->used_offload ? "xrt offload" : "not executed"));
		common::reporting::print_key_value("Trunc active comparators", trunc_stats->active_comparators);
	}
	if (ran_both) {
		common::reporting::print_key_value("Full/Trunc speedup", common::reporting::format_fixed(speedup, 3, "x"));
		common::reporting::print_key_value("Speedup interpretation",
										   (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

void print_map_reduce_debug_metrics(const common::config::Config& cfg, const std::string& device_name, 
                                    const std::string& device_bdf, bool offload_enabled,
                                    const common::topk::MapReduceRunStats& stats, 
                                    const npu::map_reduce::RunStats& npu_stats) {
    if (!cfg.debug_output) {
        return;
    }

    common::reporting::print_section_header("Debug Metrics");
    common::reporting::print_key_value("NPU device", device_name);
    common::reporting::print_key_value("NPU BDF", device_bdf);
    common::reporting::print_key_value("Offload configured", (offload_enabled ? "yes" : "no"));
    common::reporting::print_key_value("Tiles used", stats.tiles_used);
    common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
    common::reporting::print_key_value("NPU dispatches", npu_stats.layer_dispatches);
}

} // namespace npu::reporting
