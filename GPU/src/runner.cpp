#include "../include/runner.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../include/algorithm.hpp"
#include "../include/reporting.hpp"
#include "common/benchmark.hpp"
#include "common/runner.hpp"

namespace gpu::topk {

namespace {

using common::config::Algorithm;
using common::config::Config;
using common::config::DataType;

// Bitonic Hooks
template <typename T> class GpuBitonicRunnerHooks final : public common::topk::BitonicRunnerHooks<T> {
  public:
	explicit GpuBitonicRunnerHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	common::topk::BasicRunStats run(
		std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, bool is_truncated_pass
	) override {
		std::vector<T> data_backup = data;

		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats> {
				std::vector<T> temp = data_backup;
				gpu::bitonic::RunStats stats{0.0, 0, 0, 0};

				auto t0 = std::chrono::high_resolution_clock::now();
				common::energy::Scope energy_scope(common::energy::Channel::E2e);

				std::size_t final_n = temp.size();
				stats = gpu::bitonic::run_topk(temp.data(), temp.size(), final_n, layers);
				temp.resize(final_n);

				energy_scope.close();
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats>{
					elapsed_wall_ms, stats.elapsed_ms, std::move(temp), stats
				};
			}
		);

		gpu::bitonic::RunStats best_stats = best.sample.stats;
		data = std::move(best.sample.value);

		if (is_truncated_pass) {
			last_trunc_stats = best_stats;
		} else {
			last_full_stats = best_stats;
		}

		common::topk::BasicRunStats stats{};
		common::topk::fill_timing_stats(stats, best);
		stats.traffic.bytes_moved = best_stats.bytes_moved;
		stats.traffic.bytes_exact = true;

		return stats;
	}

	void print_debug_metrics(const Config& cfg, const common::topk::BitonicRunStats& stats) override {
		const gpu::bitonic::RunStats* full_run = (stats.full_run_stats != nullptr) ? &last_full_stats : nullptr;
		const gpu::bitonic::RunStats* trunc_run = (stats.trunc_run_stats != nullptr) ? &last_trunc_stats : nullptr;
		gpu::reporting::print_bitonic_debug_metrics(cfg, device_name, stats, full_run, trunc_run);
	}

  private:
	std::string device_name;
	gpu::bitonic::RunStats last_full_stats{0.0, 0, 0, 0};
	gpu::bitonic::RunStats last_trunc_stats{0.0, 0, 0, 0};
};

// MapReduce Hooks
template <typename T> class GpuMapReduceHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
	explicit GpuMapReduceHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<T> run(
		const std::vector<T>& input, const Config& cfg, common::topk::MapReduceRunStats* stats
	) override {
		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats> {
				gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};
				std::vector<T> output(cfg.k);
				auto t0 = std::chrono::high_resolution_clock::now();
				common::energy::Scope energy_scope(common::energy::Channel::E2e);

				std::size_t count = gpu::map_reduce::run_topk(
					input.data(), input.size(), cfg.k, cfg.want_max, output.data(), &map_stats
				);

				energy_scope.close();
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				output.resize(count);
				return common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats>{
					elapsed_wall_ms, map_stats.elapsed_ms, std::move(output), map_stats
				};
			}
		);

		if (stats != nullptr) {
			common::topk::fill_timing_stats(*stats, best);
			stats->tiles_used = best.sample.stats.tiles_used;
			stats->aggregated_candidates = best.sample.stats.aggregated_candidates;
			stats->traffic.bytes_moved = best.sample.stats.bytes_moved;
			stats->traffic.bytes_exact = true;
		}

		last_stats = best.sample.stats;
		return std::move(best.sample.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		gpu::reporting::print_map_reduce_debug_metrics(cfg, device_name, stats, last_stats);
	}

  private:
	std::string device_name;
	gpu::map_reduce::RunStats last_stats{0.0, 0, 0, 0};
};

// Ground Truth Hooks
template <typename T> class GpuGroundTruthHooks final : public common::topk::GroundTruthRunnerHooks<T> {
  public:
	explicit GpuGroundTruthHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<T> run(
		const std::vector<T>& input, const Config& cfg, common::topk::GroundTruthRunStats* stats
	) override {
		const std::size_t k = std::min(cfg.k, input.size());

		auto best =
			common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, double> {
				std::vector<T> temp = input;
				auto t0 = std::chrono::high_resolution_clock::now();
				common::energy::Scope energy_scope(common::energy::Channel::E2e);

				double algo_ms = gpu::ground_truth::run_topk(temp.data(), temp.size(), k, cfg.want_max);

				energy_scope.close();
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				if (k > 0 && k < temp.size()) {
					temp.resize(k);
				} else if (k == 0) {
					temp.clear();
				}

				return common::benchmark::TimedValueWithStats<std::vector<T>, double>{
					elapsed_wall_ms, algo_ms, std::move(temp), algo_ms
				};
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
	std::string device_name;
};

// Dispatch
template <typename T> int topk_typed(const Config& cfg) {
	switch (cfg.algorithm) {
	case Algorithm::MapReduce: {
		GpuMapReduceHooks<T> hooks;
		return common::topk::execute_map_reduce<T>(cfg, hooks);
	}
	case Algorithm::GroundTruth: {
		GpuGroundTruthHooks<T> hooks;
		return common::topk::execute_ground_truth<T>(cfg, hooks);
	}
	case Algorithm::Bitonic: {
		GpuBitonicRunnerHooks<T> hooks;
		return common::topk::execute_bitonic<T>(cfg, hooks);
	}
	}
	throw std::invalid_argument("Unsupported algorithm");
}

} // namespace

int execute(const common::config::Config& cfg) {
	return common::topk::dispatch_by_dtype(cfg.dtype, [&](auto tag) { return topk_typed<decltype(tag)>(cfg); });
}

} // namespace gpu::topk
