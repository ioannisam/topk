#include "../include/runner.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../include/algorithm.hpp"
#include "common/runner.hpp"
#include "../include/reporting.hpp"

namespace gpu::topk {

namespace {

using common::config::Config;
using common::config::DataType;
template <typename T> class GpuRunnerHooks final : public common::topk::RunnerHooks<T> {
  public:
	explicit GpuRunnerHooks(bool fp16_emulation)
		: device_name(gpu::bitonic::query_device_name()), use_fp16_path(fp16_emulation) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	common::topk::BasicRunStats run_network(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
										const std::vector<std::vector<unsigned char>>& keep,
										bool trunc) override {
		gpu::bitonic::RunStats stats{0.0, 0, 0, 0};
		if constexpr (std::is_same_v<T, float>) {
			if (use_fp16_path) {
				stats = gpu::bitonic::run_network_cuda_fp16(data, layers, keep, trunc);
			} else {
				stats = gpu::bitonic::run_network_cuda(data, layers, keep, trunc);
			}
		} else {
			stats = gpu::bitonic::run_network_cuda(data, layers, keep, trunc);
		}

		if (trunc) {
			last_trunc_stats = stats;
		} else {
			last_full_stats = stats;
		}
		return common::topk::BasicRunStats{stats.elapsed_ms};
	}

	void print_debug_metrics(const Config& cfg, std::size_t layer_count, std::size_t full_cmp,
						 std::size_t trunc_cmp, const common::topk::BasicRunStats* full_stats,
						 const common::topk::BasicRunStats* trunc_stats) override {
		const gpu::bitonic::RunStats* full_run = (full_stats != nullptr) ? &last_full_stats : nullptr;
		const gpu::bitonic::RunStats* trunc_run = (trunc_stats != nullptr) ? &last_trunc_stats : nullptr;
		gpu::reporting::print_debug_metrics(cfg, device_name, full_run, trunc_run, layer_count, full_cmp, trunc_cmp);
	}

  private:
	std::string device_name;
	bool use_fp16_path = false;
	gpu::bitonic::RunStats last_full_stats{0.0, 0, 0, 0};
	gpu::bitonic::RunStats last_trunc_stats{0.0, 0, 0, 0};
};

template <typename T> int topk_typed(const Config& cfg) {
	GpuRunnerHooks<T> hooks(false);
	return common::topk::execute<T>(cfg, hooks);
}

int topk_fp16(const Config& cfg) {
	GpuRunnerHooks<float> hooks(true);
	return common::topk::execute<float>(cfg, hooks);
}

} // namespace

int execute(const common::config::Config& cfg) {
	switch (cfg.dtype) {
	case DataType::Int:
		return topk_typed<std::int32_t>(cfg);
	case DataType::UInt:
		return topk_typed<std::uint32_t>(cfg);
	case DataType::Float:
		return topk_typed<float>(cfg);
	case DataType::Double:
		return topk_typed<double>(cfg);
	case DataType::Fp16:
		return topk_fp16(cfg);
	}

	throw std::invalid_argument("Unsupported dtype");
}

} // namespace gpu::topk
