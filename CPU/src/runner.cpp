#include "../include/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "../include/algorithm.hpp"
#include "../include/reporting.hpp"
#include "common/benchmark.hpp"
#include "common/runner.hpp"

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

// ==========================================
// Bitonic Hooks
// ==========================================

template <typename T> class CpuBitonicRunnerHooks final : public common::topk::BitonicRunnerHooks<T> {
  public:
	explicit CpuBitonicRunnerHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		cpu::reporting::print_configuration(cfg, context.ex_threads, n);
	}

	common::topk::BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) override {
		std::vector<T> data_backup = data;
		double bytes_moved = 0.0;

		auto best = common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValue<std::vector<T>> {
			std::vector<T> temp = data_backup;
			auto t0 = std::chrono::high_resolution_clock::now();
			common::energy::FullScope energy_scope;

			cpu::bitonic::run_topk(temp, layers, context.ex_threads, &bytes_moved);

			energy_scope.close();
			auto t1 = std::chrono::high_resolution_clock::now();
			double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
			return common::benchmark::TimedValue<std::vector<T>>{elapsed, elapsed, std::move(temp)};
		});

		data = std::move(best.sample.value);
		common::topk::BasicRunStats stats{};
		common::topk::fill_timing_stats(stats, best);
		stats.traffic.bytes_moved = bytes_moved;
		stats.traffic.bytes_exact = true;

		return stats;
	}

	void print_debug_metrics(const Config& cfg, const common::topk::BitonicRunStats& stats) override {
		cpu::reporting::print_bitonic_debug_metrics(cfg, context.hw_threads, context.ex_threads, stats);
	}

  private:
	Context context;
};

// ==========================================
// MapReduce Hooks
// ==========================================

template <typename T> class CpuMapReduceHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
	explicit CpuMapReduceHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		cpu::reporting::print_configuration(cfg, context.ex_threads, n);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::MapReduceRunStats* stats) override {
		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, cpu::map_reduce::RunStats> {
				cpu::map_reduce::RunStats run_stats{};
				auto t0 = std::chrono::high_resolution_clock::now();
				common::energy::FullScope energy_scope;

				std::vector<T> output =
					cpu::map_reduce::run_topk(input, cfg.k, cfg.want_max, context.ex_threads, &run_stats);

				energy_scope.close();
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, cpu::map_reduce::RunStats>{
					elapsed, elapsed, std::move(output), run_stats};
			});

		if (stats != nullptr) {
			common::topk::fill_timing_stats(*stats, best);
			stats->tiles_used = best.sample.stats.tiles_used;
			stats->aggregated_candidates = best.sample.stats.aggregated_candidates;
			stats->traffic.bytes_moved = best.sample.stats.bytes_moved;
			stats->traffic.bytes_exact = true;
		}

		return std::move(best.sample.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		cpu::reporting::print_map_reduce_debug_metrics(cfg, context.hw_threads, context.ex_threads, stats);
	}

  private:
	Context context;
};

// ==========================================
// Ground Truth Hooks
// ==========================================

template <typename T> class CpuGroundTruthHooks final : public common::topk::GroundTruthRunnerHooks<T> {
  public:
	explicit CpuGroundTruthHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		cpu::reporting::print_configuration(cfg, context.ex_threads, n);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::GroundTruthRunStats* stats) override {
		const std::size_t k = std::min(cfg.k, input.size());

		auto best = common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValue<std::vector<T>> {
			std::vector<T> temp = input;
			auto t0 = std::chrono::high_resolution_clock::now();
			common::energy::FullScope energy_scope;

			cpu::ground_truth::run_topk(temp, k, cfg.want_max);

			energy_scope.close();
			auto t1 = std::chrono::high_resolution_clock::now();
			double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

			if (k > 0 && k < temp.size())
				temp.resize(k);
			else if (k == 0)
				temp.clear();

			return common::benchmark::TimedValue<std::vector<T>>{elapsed_wall_ms, elapsed_wall_ms, std::move(temp)};
		});

		if (stats != nullptr) {
			common::topk::fill_timing_stats(*stats, best);
			stats->traffic.bytes_moved = static_cast<double>(input.size() + k) * static_cast<double>(sizeof(T));
		}

		return std::move(best.sample.value);
	}

	void print_debug_metrics(const Config&, const common::topk::GroundTruthRunStats&) override {
		// no specific GT metrics to output
	}

  private:
	Context context;
};

// ==========================================
// Dispatch
// ==========================================

template <typename T> int topk_typed(const Config& cfg) {
	const Context ctx = build_context(cfg);

	if (cfg.algorithm == Algorithm::MapReduce) {
		CpuMapReduceHooks<T> hooks(ctx);
		return common::topk::execute_map_reduce<T>(cfg, hooks);
	} else if (cfg.algorithm == Algorithm::GroundTruth) {
		CpuGroundTruthHooks<T> hooks(ctx);
		return common::topk::execute_ground_truth<T>(cfg, hooks);
	}

	CpuBitonicRunnerHooks<T> hooks(ctx);
	return common::topk::execute_bitonic<T>(cfg, hooks);
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
	case DataType::Half:
#if defined(__FLT16_MANT_DIG__)
		return topk_typed<_Float16>(cfg);
#else
		throw std::invalid_argument("dtype=half is not supported by this compiler target");
#endif
	}

	throw std::invalid_argument("Unsupported dtype");
}

} // namespace cpu::topk
