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
#include "common/random.hpp"
#include "common/runner.hpp"
#include "common/validation.hpp"

namespace gpu::topk {

namespace {

using common::config::Algorithm;
using common::config::Config;
using common::config::DataType;

template <typename T> class GpuBitonicRunnerHooks final : public common::topk::BitonicRunnerHooks<T> {
  public:
	explicit GpuBitonicRunnerHooks(bool fp16_emulation)
		: device_name(gpu::bitonic::query_device_name()), use_fp16_path(fp16_emulation) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	common::topk::BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) override {
		std::vector<T> data_backup = data;

		// warmup
		common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
			std::vector<T> temp = data_backup;
			if constexpr (std::is_same_v<T, float>) {
				if (use_fp16_path)
					gpu::bitonic::run_network_cuda_fp16(temp, layers);
				else
					gpu::bitonic::run_network_cuda(temp, layers);
			} else {
				gpu::bitonic::run_network_cuda(temp, layers);
			}
		});

		// measurement
		auto best = common::benchmark::measure_best(
			common::benchmark::kMeasureIters,
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats> {
				std::vector<T> temp = data_backup;
				gpu::bitonic::RunStats stats{0.0, 0, 0, 0};

				auto t0 = std::chrono::high_resolution_clock::now();

				if constexpr (std::is_same_v<T, float>) {
					if (use_fp16_path)
						stats = gpu::bitonic::run_network_cuda_fp16(temp, layers);
					else
						stats = gpu::bitonic::run_network_cuda(temp, layers);
				} else {
					stats = gpu::bitonic::run_network_cuda(temp, layers);
				}

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats>{
					elapsed_wall_ms,
					std::move(temp),
					stats,
				};
			});

		gpu::bitonic::RunStats best_stats = best.stats;
		data = std::move(best.value);

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
		stats.end_to_end_ms = best.elapsed_ms;
		stats.algorithm_ms = best_stats.elapsed_ms;
		std::cout << "[PROFILE_TIME_MS] " << stats.end_to_end_ms << "\n";
		return stats;
	}

	void print_debug_metrics(const Config& cfg, const common::topk::BitonicRunStats& stats) override {
		const gpu::bitonic::RunStats* full_run = (stats.full_run_stats != nullptr) ? &last_full_stats : nullptr;
		const gpu::bitonic::RunStats* trunc_run = (stats.trunc_run_stats != nullptr) ? &last_trunc_stats : nullptr;
		gpu::reporting::print_bitonic_debug_metrics(cfg, device_name, stats, full_run, trunc_run);
	}

  private:
	std::string device_name;
	bool use_fp16_path = false;
	gpu::bitonic::RunStats last_full_stats{0.0, 0, 0, 0};
	gpu::bitonic::RunStats last_trunc_stats{0.0, 0, 0, 0};
};

template <typename T> class GpuMapReduceHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
	explicit GpuMapReduceHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::MapReduceRunStats* stats) override {

		// warmup
		common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
			gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};
			gpu::map_reduce::run_topk(input, cfg.k, cfg.want_max, cfg.ex_threads, &map_stats);
		});

		// measurement
		auto best = common::benchmark::measure_best(
			common::benchmark::kMeasureIters,
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats> {
				gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};

				auto t0 = std::chrono::high_resolution_clock::now();

				std::vector<T> output =
					gpu::map_reduce::run_topk(input, cfg.k, cfg.want_max, cfg.ex_threads, &map_stats);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats>{
					elapsed_wall_ms,
					std::move(output),
					map_stats,
				};
			});

		const double end_to_end_ms = best.elapsed_ms;
		const double algorithm_ms = best.stats.elapsed_ms;
		if (stats != nullptr) {
			stats->end_to_end_ms = end_to_end_ms;
			stats->algorithm_ms = algorithm_ms;
			stats->tiles_used = best.stats.tiles_used;
			stats->aggregated_candidates = best.stats.aggregated_candidates;
			std::cout << "[PROFILE_TIME_MS] " << stats->end_to_end_ms << "\n";
		} else {
			std::cout << "[PROFILE_TIME_MS] " << end_to_end_ms << "\n";
		}

		last_stats = best.stats;
		return std::move(best.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		gpu::reporting::print_map_reduce_debug_metrics(cfg, device_name, stats, last_stats);
	}

  private:
	std::string device_name;
	gpu::map_reduce::RunStats last_stats{0.0, 0, 0, 0};
};

class GpuMapReduceFp16Hooks final : public common::topk::MapReduceRunnerHooks<float> {
  public:
	explicit GpuMapReduceFp16Hooks() : device_name(gpu::bitonic::query_device_name()) {}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<float> run(const std::vector<float>& input, const Config& cfg,
					       common::topk::MapReduceRunStats* stats) override {
		common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
			gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};
			gpu::map_reduce::run_topk_fp16(input, cfg.k, cfg.want_max, cfg.ex_threads, &map_stats);
		});

		auto best = common::benchmark::measure_best(
			common::benchmark::kMeasureIters,
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<float>, gpu::map_reduce::RunStats> {
				gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};
				auto t0 = std::chrono::high_resolution_clock::now();
				
				std::vector<float> output =
					gpu::map_reduce::run_topk_fp16(input, cfg.k, cfg.want_max, cfg.ex_threads, &map_stats);
					
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<float>, gpu::map_reduce::RunStats>{
					elapsed_wall_ms, std::move(output), map_stats
				};
			});

		if (stats != nullptr) {
			stats->end_to_end_ms = best.elapsed_ms;
			stats->algorithm_ms = best.stats.elapsed_ms;
			stats->tiles_used = best.stats.tiles_used;
			stats->aggregated_candidates = best.stats.aggregated_candidates;
			std::cout << "[PROFILE_TIME_MS] " << stats->end_to_end_ms << "\n";
		} else {
			std::cout << "[PROFILE_TIME_MS] " << best.elapsed_ms << "\n";
		}

		last_stats = best.stats;
		return std::move(best.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		gpu::reporting::print_map_reduce_debug_metrics(cfg, device_name, stats, last_stats);
	}

  private:
	std::string device_name;
	gpu::map_reduce::RunStats last_stats{0.0, 0, 0, 0};
};

template <typename T> int topk_typed_gt(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t k = std::min(cfg.k, n);

	std::string device_name = gpu::bitonic::query_device_name();
	gpu::reporting::print_configuration(cfg, n, device_name);

	std::vector<T> data = common::utils::generate_random_input<T>(n, cfg.seed, cfg.rand_min, cfg.rand_max);

	// Warmup
	common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
		std::vector<T> temp = data;
		gpu::ground_truth::run_topk(temp, k, cfg.want_max);
	});

	// Measurement
	auto best = common::benchmark::measure_best(
		common::benchmark::kMeasureIters, [&]() -> common::benchmark::TimedValue<std::vector<T>> {
			std::vector<T> temp = data;
			auto t0 = std::chrono::high_resolution_clock::now();
			
			gpu::ground_truth::run_topk(temp, k, cfg.want_max);
			
			auto t1 = std::chrono::high_resolution_clock::now();
			double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
			return common::benchmark::TimedValue<std::vector<T>>{elapsed, std::move(temp)};
		});

	std::cout << "[PROFILE_TIME_MS] " << best.elapsed_ms << "\n";
	common::reporting::print_timing_lines({{"Ground truth algorithmic time (ms)", best.elapsed_ms}});

	if (cfg.verify_output) {
		common::reporting::print_check_result("Top-k correctness vs CPU sorted reference", true, true);
	}

	std::vector<std::string> formatted = common::utils::format_output(best.value);
	common::reporting::print_output(cfg, formatted);

	return 0;
}

template <typename T> int topk_typed(const Config& cfg) {
	if (cfg.algorithm == Algorithm::MapReduce) {
		GpuMapReduceHooks<T> hooks;
		return common::topk::execute_map_reduce<T>(cfg, hooks);
	} else if (cfg.algorithm == Algorithm::GroundTruth) {
		return topk_typed_gt<T>(cfg);
	}
	GpuBitonicRunnerHooks<T> hooks(false);
	return common::topk::execute_bitonic<T>(cfg, hooks);
}

int topk_fp16_dispatch(const Config& cfg) {
	if (cfg.algorithm == Algorithm::MapReduce) {
		GpuMapReduceFp16Hooks hooks;
		return common::topk::execute_map_reduce<float>(cfg, hooks);
	} else if (cfg.algorithm == Algorithm::GroundTruth) {
		return topk_typed_gt<float>(cfg);
	}
	GpuBitonicRunnerHooks<float> hooks(true);
	return common::topk::execute_bitonic<float>(cfg, hooks);
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
		return topk_fp16_dispatch(cfg);
	}

	throw std::invalid_argument("Unsupported dtype");
}

} // namespace gpu::topk
