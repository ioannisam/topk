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

using common::config::Algorithm;
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

template <typename T> class CpuBitonicRunnerHooks final : public common::topk::BitonicRunnerHooks<T> {
  public:
	explicit CpuBitonicRunnerHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		cpu::reporting::print_configuration(cfg, context.ex_threads, n);
	}

	common::topk::BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) override {
		auto t0 = std::chrono::high_resolution_clock::now();
		cpu::bitonic::run_topk(data, layers, context.ex_threads);
		auto t1 = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
		std::cout << "[PROFILE_TIME_MS] " << elapsed << "\n";
		return common::topk::BasicRunStats{elapsed};
	}

	void print_debug_metrics(const Config& cfg, std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp,
							 const common::topk::BasicRunStats* full_stats,
							 const common::topk::BasicRunStats* trunc_stats) override {
		cpu::reporting::print_debug_metrics(cfg, context.hw_threads, context.ex_threads, layer_count, full_cmp,
											trunc_cmp, full_stats != nullptr ? full_stats->elapsed_ms : 0.0,
											trunc_stats != nullptr ? trunc_stats->elapsed_ms : 0.0);
	}

  private:
	Context context;
};

template <typename T> class CpuMapReduceHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
	explicit CpuMapReduceHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		cpu::reporting::print_configuration(cfg, context.ex_threads, n);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::MapReduceRunStats* stats) override {
		cpu::map_reduce::RunStats map_stats{};
		auto t0 = std::chrono::high_resolution_clock::now();
		std::vector<T> output = cpu::map_reduce::run_topk(input, cfg.k, cfg.want_max, context.ex_threads, &map_stats);
		auto t1 = std::chrono::high_resolution_clock::now();

		double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
		if (stats != nullptr) {
			stats->elapsed_ms = elapsed;
			stats->tiles_used = map_stats.tiles_used;
			stats->aggregated_candidates = map_stats.aggregated_candidates;
			std::cout << "[PROFILE_TIME_MS] " << stats->elapsed_ms << "\n";
		} else {
			std::cout << "[PROFILE_TIME_MS] " << elapsed << "\n";
		}

		return output;
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		if (!cfg.debug_output) {
			return;
		}

		common::reporting::print_section_header("Debug Metrics");
		common::reporting::print_key_value("Hardware threads available", context.hw_threads);
		common::reporting::print_key_value("Execution threads used", context.ex_threads);
		common::reporting::print_key_value("Tiles used", stats.tiles_used);
		common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
	}

  private:
	Context context;
};

template <typename T> int topk_typed_bitonic(const Config& cfg) {
	const Context ctx = build_context(cfg);
	CpuBitonicRunnerHooks<T> hooks(ctx);
	return common::topk::execute_bitonic<T>(cfg, hooks);
}

template <typename T> int topk_typed_map_reduce(const Config& cfg) {
	const Context ctx = build_context(cfg);
	CpuMapReduceHooks<T> hooks(ctx);
	return common::topk::execute_map_reduce<T>(cfg, hooks);
}

template <typename T> int topk_typed(const Config& cfg) {
	if (cfg.algorithm == Algorithm::MapReduce) {
		return topk_typed_map_reduce<T>(cfg);
	}
	return topk_typed_bitonic<T>(cfg);
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
