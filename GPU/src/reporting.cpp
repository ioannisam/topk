#include "../include/reporting.hpp"

#include "common/reporting.hpp"

namespace gpu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, const std::string& device_name) {
	common::reporting::print_configuration(cfg, n, std::nullopt, "Network run mode", "gpu");
	common::reporting::print_key_value("CUDA device", device_name);
}

void print_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
						 const gpu::bitonic::RunStats* full_stats, const gpu::bitonic::RunStats* trunc_stats,
						 std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp) {
	if (!cfg.debug_output) {
		return;
	}

	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("CUDA device", device_name);
	common::reporting::print_key_value("Bitonic layers", layer_count);
	common::reporting::print_key_value("Full comparators", full_cmp);
	common::reporting::print_key_value("Trunc comparators", trunc_cmp);
	common::reporting::print_key_value("Comparator skip ratio", common::reporting::format_fixed(skipped_pct, 2, "%"));

	if (full_stats != nullptr) {
		common::reporting::print_key_value("Full kernels launched", full_stats->kernel_launches);
		common::reporting::print_key_value("CUDA block size", full_stats->block_size);
	}
	if (trunc_stats != nullptr) {
		common::reporting::print_key_value("Trunc kernels launched", trunc_stats->kernel_launches);
		common::reporting::print_key_value("Trunc active comparators", trunc_stats->active_comparators);
	}
	if (full_stats != nullptr && trunc_stats != nullptr && trunc_stats->elapsed_ms > 0.0) {
		const double speedup = full_stats->elapsed_ms / trunc_stats->elapsed_ms;
		common::reporting::print_key_value("Full/Trunc speedup", common::reporting::format_fixed(speedup, 3, "x"));
		common::reporting::print_key_value("Speedup interpretation",
										   (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

} // namespace gpu::reporting
