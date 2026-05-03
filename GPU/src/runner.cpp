#include "../include/runner.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../include/algorithm.hpp"
#include "common/runner.hpp"
#include "common/benchmark.hpp"
#include "../include/reporting.hpp"

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
				if (use_fp16_path) gpu::bitonic::run_network_cuda_fp16(temp, layers);
				else gpu::bitonic::run_network_cuda(temp, layers);
			} else {
				gpu::bitonic::run_network_cuda(temp, layers);
			}
		});

		// measurement
		auto best = common::benchmark::measure_best(common::benchmark::kMeasureIters, [&]()
			-> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats> {
			std::vector<T> temp = data_backup;
			gpu::bitonic::RunStats stats{0.0, 0, 0, 0};
			
			auto t0 = std::chrono::high_resolution_clock::now();

			if constexpr (std::is_same_v<T, float>) {
				if (use_fp16_path) stats = gpu::bitonic::run_network_cuda_fp16(temp, layers);
				else stats = gpu::bitonic::run_network_cuda(temp, layers);
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
		std::cout << "[PROFILE_TIME_MS] " << best.elapsed_ms << "\n";
		return common::topk::BasicRunStats{best.elapsed_ms};
	}

	void print_debug_metrics(const Config& cfg, std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp,
							 const common::topk::BasicRunStats* full_stats,
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
		auto best = common::benchmark::measure_best(common::benchmark::kMeasureIters, [&]()
			-> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats> {
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

		if (stats != nullptr) {
			stats->elapsed_ms = best.elapsed_ms;
			stats->tiles_used = best.stats.tiles_used;
			stats->aggregated_candidates = best.stats.aggregated_candidates;
			std::cout << "[PROFILE_TIME_MS] " << stats->elapsed_ms << "\n";
		} else {
			std::cout << "[PROFILE_TIME_MS] " << best.elapsed_ms << "\n";
		}

		last_stats = best.stats;
		return std::move(best.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
		if (!cfg.debug_output) {
			return;
		}

		common::reporting::print_section_header("Debug Metrics");
		common::reporting::print_key_value("CUDA device", device_name);
		common::reporting::print_key_value("Tiles used", stats.tiles_used);
		common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
		common::reporting::print_key_value("CUDA block size", last_stats.block_size);
	}

  private:
	std::string device_name;
	gpu::map_reduce::RunStats last_stats{0.0, 0, 0, 0};
};

template <typename T> int topk_typed(const Config& cfg) {
	GpuBitonicRunnerHooks<T> hooks(false);
	return common::topk::execute_bitonic<T>(cfg, hooks);
}

template <typename T> int topk_typed_map_reduce(const Config& cfg) {
	GpuMapReduceHooks<T> hooks;
	return common::topk::execute_map_reduce<T>(cfg, hooks);
}

int topk_fp16(const Config& cfg) {
	GpuBitonicRunnerHooks<float> hooks(true);
	return common::topk::execute_bitonic<float>(cfg, hooks);
}

} // namespace

int execute(const common::config::Config& cfg) {
	switch (cfg.dtype) {
	case DataType::Int:
		return cfg.algorithm == Algorithm::MapReduce ? topk_typed_map_reduce<std::int32_t>(cfg)
													 : topk_typed<std::int32_t>(cfg);
	case DataType::UInt:
		return cfg.algorithm == Algorithm::MapReduce ? topk_typed_map_reduce<std::uint32_t>(cfg)
													 : topk_typed<std::uint32_t>(cfg);
	case DataType::Float:
		return cfg.algorithm == Algorithm::MapReduce ? topk_typed_map_reduce<float>(cfg) : topk_typed<float>(cfg);
	case DataType::Double:
		return cfg.algorithm == Algorithm::MapReduce ? topk_typed_map_reduce<double>(cfg) : topk_typed<double>(cfg);
	case DataType::Fp16:
		if (cfg.algorithm == Algorithm::MapReduce) {
			throw std::invalid_argument("GPU map-reduce currently supports dtype=int|uint|float|double");
		}
		return topk_fp16(cfg);
	}

	throw std::invalid_argument("Unsupported dtype");
}

} // namespace gpu::topk
