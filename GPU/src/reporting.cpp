#include "../include/reporting.hpp"

#include "common/reporting.hpp"

namespace gpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, const std::string& device_name) {
	common::reporting::print_configuration(cfg, n, std::nullopt, "Network run mode", "gpu");
	common::reporting::print_key_value("CUDA device", device_name);
}

void print_bitonic_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
                                 const common::topk::BitonicRunStats& stats,
                                 const gpu::bitonic::RunStats* full_gpu_stats,
                                 const gpu::bitonic::RunStats* trunc_gpu_stats) {
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
	common::reporting::print_key_value("CUDA device", device_name);
	common::reporting::print_key_value("Bitonic layers", stats.layer_count);
	common::reporting::print_key_value("Full comparators", full_cmp);
	common::reporting::print_key_value("Trunc comparators", trunc_cmp);
	common::reporting::print_key_value("Comparator skip ratio", common::reporting::format_fixed(skipped_pct, 2, "%"));

	if (full_gpu_stats != nullptr) {
		common::reporting::print_key_value("Full kernels launched", full_gpu_stats->kernel_launches);
		common::reporting::print_key_value("CUDA block size", full_gpu_stats->block_size);
	}
	if (trunc_gpu_stats != nullptr) {
		common::reporting::print_key_value("Trunc kernels launched", trunc_gpu_stats->kernel_launches);
		common::reporting::print_key_value("Trunc active comparators", trunc_gpu_stats->active_comparators);
	}
	if (ran_both) {
		common::reporting::print_key_value("Full/Trunc speedup", common::reporting::format_fixed(speedup, 3, "x"));
		common::reporting::print_key_value("Speedup interpretation",
										   (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

void print_map_reduce_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
                                    const common::topk::MapReduceRunStats& stats,
                                    const gpu::map_reduce::RunStats& gpu_stats) {
	if (!cfg.debug_output) {
		return;
	}

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("CUDA device", device_name);
	common::reporting::print_key_value("Tiles used", stats.tiles_used);
	common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
	common::reporting::print_key_value("CUDA block size", gpu_stats.block_size);
}

} // namespace gpu::reporting
