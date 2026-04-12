#include "../include/reporting.hpp"

#include "common/reporting.hpp"

namespace npu::reporting {

void print_configuration(const common::config::Config& cfg, std::size_t n, std::size_t ex_threads,
					 const std::string& device_name, const std::string& device_bdf,
					 bool offload_enabled) {
	common::reporting::print_configuration(cfg, n, ex_threads, "Network run mode", "npu");
	common::reporting::print_key_value("NPU device", device_name);
	common::reporting::print_key_value("NPU BDF", device_bdf);
	common::reporting::print_key_value("Offload configured", (offload_enabled ? "yes" : "no"));
}

void print_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
				 const npu::bitonic::RunStats* full_stats, const npu::bitonic::RunStats* trunc_stats,
				 std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp) {
	if (!cfg.debug_output) {
		return;
	}

	const double skipped_pct =
		full_cmp == 0 ? 0.0 : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));

	common::reporting::print_section_header("Debug Metrics");
	common::reporting::print_key_value("NPU device", device_name);
	common::reporting::print_key_value("Bitonic layers", layer_count);
	common::reporting::print_key_value("Full comparators", full_cmp);
	common::reporting::print_key_value("Trunc comparators", trunc_cmp);
	common::reporting::print_key_value("Comparator skip ratio", common::reporting::format_fixed(skipped_pct, 2, "%"));

	if (full_stats != nullptr) {
		common::reporting::print_key_value("Full NPU dispatches", full_stats->layer_dispatches);
		common::reporting::print_key_value("Full execution path", (full_stats->used_offload ? "xrt offload" : "not executed"));
		common::reporting::print_key_value("Execution workers", full_stats->workers);
	}
	if (trunc_stats != nullptr) {
		common::reporting::print_key_value("Trunc NPU dispatches", trunc_stats->layer_dispatches);
		common::reporting::print_key_value("Trunc execution path", (trunc_stats->used_offload ? "xrt offload" : "not executed"));
		common::reporting::print_key_value("Trunc active comparators", trunc_stats->active_comparators);
	}
	if (full_stats != nullptr && trunc_stats != nullptr && trunc_stats->elapsed_ms > 0.0) {
		const double speedup = full_stats->elapsed_ms / trunc_stats->elapsed_ms;
		common::reporting::print_key_value("Full/Trunc speedup", common::reporting::format_fixed(speedup, 3, "x"));
		common::reporting::print_key_value("Speedup interpretation", (speedup >= 1.0 ? "trunc faster" : "trunc slower"));
	}
}

} // namespace npu::reporting
