#include "../include/runner.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../include/algorithm.hpp"
#include "common/runner.hpp"
#include "../include/reporting.hpp"

namespace cpu::topk {

namespace {

using common::config::Config;
using common::config::DataType;

struct Context {
	std::size_t hw_threads;
	std::size_t ex_threads;
};

Context build_context(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
	std::size_t ex_threads = cfg.ex_threads == 0 ? hw_threads : cfg.ex_threads;
	ex_threads = std::max<std::size_t>(1, std::min(ex_threads, n));

	return Context{hw_threads, ex_threads};
}

template <typename T> class CpuRunnerHooks final : public common::topk::RunnerHooks<T> {
  public:
	explicit CpuRunnerHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		cpu::reporting::print_configuration(cfg, context.ex_threads, n);
	}

	common::topk::BasicRunStats run_network(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
										const std::vector<std::vector<unsigned char>>& keep,
										bool trunc) override {
		auto t0 = std::chrono::high_resolution_clock::now();
		cpu::bitonic::run_network_parallel(data, layers, keep, trunc, context.ex_threads);
		auto t1 = std::chrono::high_resolution_clock::now();
		return common::topk::BasicRunStats{std::chrono::duration<double, std::milli>(t1 - t0).count()};
	}

	void print_debug_metrics(const Config& cfg, std::size_t layer_count, std::size_t full_cmp,
						 std::size_t trunc_cmp, const common::topk::BasicRunStats* full_stats,
						 const common::topk::BasicRunStats* trunc_stats) override {
		cpu::reporting::print_debug_metrics(cfg, context.hw_threads, context.ex_threads, layer_count, full_cmp, trunc_cmp,
						full_stats != nullptr ? full_stats->elapsed_ms : 0.0,
						trunc_stats != nullptr ? trunc_stats->elapsed_ms : 0.0);
	}

  private:
	Context context;
};

template <typename T> int topk_typed(const Config& cfg) {
	const Context ctx = build_context(cfg);
	CpuRunnerHooks<T> hooks(ctx);
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

} // namespace cpu::topk
