#include "../include/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../include/algorithm.hpp"
#include "../include/reporting.hpp"
#include "common/benchmark.hpp"
#include "common/runner.hpp"

namespace npu::topk {

namespace {

using common::config::Algorithm;
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

template <typename T> class NpuBitonicRunnerHooks final : public common::topk::BitonicRunnerHooks<T> {
  public:
	explicit NpuBitonicRunnerHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		npu::reporting::print_configuration(cfg, n, context.ex_threads, context.device_name, context.device_bdf,
											context.offload_enabled);
	}

	common::topk::BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) override {
		std::vector<T> data_backup = data;

		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats> {
				std::vector<T> temp = data_backup;

				auto t0 = std::chrono::high_resolution_clock::now();
				common::energy::Scope energy_scope(common::energy::Channel::E2e);

				const npu::bitonic::RunStats stats = npu::bitonic::run_topk(temp, layers);

				energy_scope.close();
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats>{
					elapsed_wall_ms,
					stats.elapsed_ms,
					std::move(temp),
					stats,
				};
			});

		npu::bitonic::RunStats best_stats = best.sample.stats;
		data = std::move(best.sample.value);

		bool is_trunc = false;
		for (const auto& l : layers) {
			if (l.type == common::bitonic::LayerType::Truncate) {
				is_trunc = true;
				break;
			}
		}

		if (is_trunc) {
			last_trunc_stats = best_stats;
		} else {
			last_full_stats = best_stats;
		}
		common::topk::BasicRunStats stats{};
		common::topk::fill_timing_stats(stats, best);
		stats.traffic.bytes_moved = best_stats.bytes_moved;
		stats.traffic.bytes_exact = true;
		stats.traffic.compare_ops = static_cast<double>(best_stats.active_comparators);
		stats.traffic.ops_exact = true;
		return stats;
	}

	void print_debug_metrics(const Config& cfg, const common::topk::BitonicRunStats& stats) override {
		const npu::bitonic::RunStats* full_run = (stats.full_run_stats != nullptr) ? &last_full_stats : nullptr;
		const npu::bitonic::RunStats* trunc_run = (stats.trunc_run_stats != nullptr) ? &last_trunc_stats : nullptr;
		npu::reporting::print_bitonic_debug_metrics(cfg, context.device_name, stats, full_run, trunc_run);
	}

  private:
	Context context;
	npu::bitonic::RunStats last_full_stats{0.0, 0, 0, 0, false};
	npu::bitonic::RunStats last_trunc_stats{0.0, 0, 0, 0, false};
};

template <typename T> class NpuMapReduceRunnerHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
	explicit NpuMapReduceRunnerHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		npu::reporting::print_configuration(cfg, n, context.ex_threads, context.device_name, context.device_bdf,
											context.offload_enabled);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::MapReduceRunStats* stats) override {
		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, npu::map_reduce::RunStats> {
				npu::map_reduce::RunStats run_stats{0.0, 0, false};
				auto t0 = std::chrono::high_resolution_clock::now();
				common::energy::Scope energy_scope(common::energy::Channel::E2e);

				std::vector<T> output = npu::map_reduce::run_topk(input, cfg.k, cfg.want_max, &run_stats);

				energy_scope.close();
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, npu::map_reduce::RunStats>{
					elapsed_wall_ms,
					run_stats.elapsed_ms,
					std::move(output),
					run_stats,
				};
			});

		last_run_stats = best.sample.stats;
		if (stats != nullptr) {
			common::topk::fill_timing_stats(*stats, best);
			stats->tiles_used = 0;
			stats->aggregated_candidates = 0;
			stats->traffic.bytes_moved = best.sample.stats.bytes_moved;
			stats->traffic.bytes_exact = true;
		}

		return std::move(best.sample.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		npu::reporting::print_map_reduce_debug_metrics(cfg, context.device_name, context.device_bdf,
													   context.offload_enabled, stats, last_run_stats);
	}

  private:
	Context context;
	npu::map_reduce::RunStats last_run_stats{0.0, 0, false};
};

template <typename T> class NpuGroundTruthHooks final : public common::topk::GroundTruthRunnerHooks<T> {
  public:
	explicit NpuGroundTruthHooks(const Context& ctx) : context(ctx) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		npu::reporting::print_configuration(cfg, n, context.ex_threads, context.device_name, context.device_bdf,
											context.offload_enabled);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::GroundTruthRunStats* stats) override {
		const std::size_t k = std::min(cfg.k, input.size());

		auto best = common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValue<std::vector<T>> {
			std::vector<T> temp = input;
			auto t0 = std::chrono::high_resolution_clock::now();
			common::energy::FullScope energy_scope;

			npu::ground_truth::run_topk(temp, k, cfg.want_max);

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

template <typename T> int topk_typed(const Config& cfg) {
	const Context ctx = build_context(cfg);

	if (cfg.algorithm == Algorithm::Bitonic) {
		NpuBitonicRunnerHooks<T> hooks(ctx);
		return common::topk::execute_bitonic<T>(cfg, hooks);
	}
	if (cfg.algorithm == Algorithm::GroundTruth) {
		NpuGroundTruthHooks<T> hooks(ctx);
		return common::topk::execute_ground_truth<T>(cfg, hooks);
	}
	NpuMapReduceRunnerHooks<T> hooks(ctx);
	return common::topk::execute_map_reduce<T>(cfg, hooks);
}

} // namespace

int execute(const common::config::Config& cfg) {
	return common::topk::dispatch_by_dtype(cfg.dtype, [&](auto tag) { return topk_typed<decltype(tag)>(cfg); });
}

} // namespace npu::topk
