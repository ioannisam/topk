#include "../include/runner.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../include/algorithm.hpp"
#include "../include/reporting.hpp"
#include "common/runner.hpp"

namespace npu::topk {

namespace {

using common::config::Config;
using common::config::DataType;

struct Context {
	std::size_t hw_threads;
	std::size_t ex_threads;
	std::string device_name;
	std::string device_bdf;
	bool offload_enabled;
};

Context build_context(const Config& cfg) {
	(void)cfg;
	const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
	const std::size_t ex_threads = 1;
	return Context{hw_threads, ex_threads, npu::bitonic::query_device_name(), npu::bitonic::query_device_bdf(),
				   npu::bitonic::is_offload_configured()};
}

template <typename T> class NpuRunnerHooks final : public common::topk::RunnerHooks<T> {
  public:
	explicit NpuRunnerHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		npu::reporting::print_configuration(cfg, n, context.ex_threads, context.device_name, context.device_bdf,
							   context.offload_enabled);
	}

	common::topk::BasicRunStats run_network(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
										const std::vector<std::vector<unsigned char>>& keep,
										bool trunc) override {
		const npu::bitonic::RunStats stats = npu::bitonic::run_network_npu(data, layers, keep, trunc, context.ex_threads);
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
		const npu::bitonic::RunStats* full_run = (full_stats != nullptr) ? &last_full_stats : nullptr;
		const npu::bitonic::RunStats* trunc_run = (trunc_stats != nullptr) ? &last_trunc_stats : nullptr;
		npu::reporting::print_debug_metrics(cfg, context.device_name, full_run, trunc_run, layer_count, full_cmp, trunc_cmp);
	}

  private:
	Context context;
	npu::bitonic::RunStats last_full_stats{0.0, 0, 0, 0, false};
	npu::bitonic::RunStats last_trunc_stats{0.0, 0, 0, 0, false};
};

template <typename T> int topk_typed(const Config& cfg) {
	const Context ctx = build_context(cfg);
	NpuRunnerHooks<T> hooks(ctx);
	return common::topk::execute<T>(cfg, hooks);
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
#if defined(__FLT16_MANT_DIG__)
		return topk_typed<_Float16>(cfg);
#else
		throw std::invalid_argument("dtype=fp16 is not supported by this compiler target");
#endif
	}

	throw std::invalid_argument("Unsupported dtype");
}

} // namespace npu::topk
