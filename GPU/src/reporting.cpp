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

	common::reporting::print_bitonic_common_metrics(cfg, stats);
	common::reporting::print_key_value("CUDA device", device_name);

	if (full_gpu_stats != nullptr) {
		common::reporting::print_key_value("Full kernels launched", full_gpu_stats->kernel_launches);
		common::reporting::print_key_value("CUDA block size", full_gpu_stats->block_size);
	}
	if (trunc_gpu_stats != nullptr) {
		common::reporting::print_key_value("Trunc kernels launched", trunc_gpu_stats->kernel_launches);
		common::reporting::print_key_value("Trunc active comparators", trunc_gpu_stats->active_comparators);
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
