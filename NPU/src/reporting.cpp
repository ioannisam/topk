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

namespace {

void print_phase_lines(const char* prefix, const npu::PhaseTimers& phases) {
	const std::string label = prefix;
	common::reporting::print_key_value((label + " phase sample (ms)").c_str(), phases.sample_ms, 3);
	common::reporting::print_key_value((label + " phase setup (ms)").c_str(), phases.setup_ms, 3);
	common::reporting::print_key_value((label + " phase stage (ms)").c_str(), phases.stage_ms, 3);
	common::reporting::print_key_value((label + " phase dispatch (ms)").c_str(), phases.dispatch_ms, 3);
	common::reporting::print_key_value((label + " phase wait (ms)").c_str(), phases.wait_ms, 3);
	common::reporting::print_key_value((label + " phase merge (ms)").c_str(), phases.merge_ms, 3);
	common::reporting::print_key_value((label + " phase finalize (ms)").c_str(), phases.finalize_ms, 3);
}

} // namespace

void print_bitonic_debug_metrics(const common::config::Config& cfg, const std::string& device_name,
								 const common::topk::BitonicRunStats& stats, const npu::bitonic::RunStats* full_stats,
								 const npu::bitonic::RunStats* trunc_stats) {
	if (!cfg.debug_output) {
		return;
	}

	common::reporting::print_bitonic_common_metrics(cfg, stats);
	common::reporting::print_key_value("NPU device", device_name);

	if (full_stats != nullptr) {
		common::reporting::print_key_value("Full NPU dispatches", full_stats->layer_dispatches);
		common::reporting::print_key_value("Full execution path",
										   (full_stats->used_offload ? "xrt offload" : "not executed"));
		common::reporting::print_key_value("Execution workers", full_stats->workers);
		print_phase_lines("Full", full_stats->phases);
	}
	if (trunc_stats != nullptr) {
		common::reporting::print_key_value("Trunc NPU dispatches", trunc_stats->layer_dispatches);
		common::reporting::print_key_value("Trunc execution path",
										   (trunc_stats->used_offload ? "xrt offload" : "not executed"));
		common::reporting::print_key_value("Trunc active comparators", trunc_stats->active_comparators);
		print_phase_lines("Trunc", trunc_stats->phases);
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
	print_phase_lines("Map-reduce", npu_stats.phases);
}

} // namespace npu::reporting
