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

#if defined(__FLT16_MANT_DIG__)
std::vector<float> to_float(const std::vector<_Float16>& in) {
	std::vector<float> out(in.size());
	for (std::size_t i = 0; i < in.size(); ++i) {
		out[i] = static_cast<float>(in[i]);
	}
	return out;
}

std::vector<_Float16> to_half(const std::vector<float>& in) {
	std::vector<_Float16> out(in.size());
	for (std::size_t i = 0; i < in.size(); ++i) {
		out[i] = static_cast<_Float16>(in[i]);
	}
	return out;
}
#endif

// ==========================================
// Bitonic Hooks
// ==========================================

template <typename T> class GpuBitonicRunnerHooks final : public common::topk::BitonicRunnerHooks<T> {
  public:
	explicit GpuBitonicRunnerHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	common::topk::BasicRunStats run(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers) override {
		std::vector<T> data_backup = data;

		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats> {
				std::vector<T> temp = data_backup;
				gpu::bitonic::RunStats stats{0.0, 0, 0, 0};

				auto t0 = std::chrono::high_resolution_clock::now();

				std::size_t final_n = temp.size();
				stats = gpu::bitonic::run_network_cuda(temp.data(), temp.size(), final_n, layers);
				temp.resize(final_n);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, gpu::bitonic::RunStats>{
					elapsed_wall_ms, std::move(temp), stats};
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

#if defined(__FLT16_MANT_DIG__)
class GpuBitonicFp16Hooks final : public common::topk::BitonicRunnerHooks<_Float16> {
  public:
	explicit GpuBitonicFp16Hooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	common::topk::BasicRunStats run(std::vector<_Float16>& data,
									const std::vector<common::bitonic::Layer>& layers) override {
		std::vector<_Float16> data_backup = data;

		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<_Float16>, gpu::bitonic::RunStats> {
				std::vector<_Float16> temp = data_backup;

				auto t0 = std::chrono::high_resolution_clock::now();
				std::size_t final_n = temp.size();
				gpu::bitonic::RunStats stats =
					gpu::bitonic::run_network_cuda_fp16(temp.data(), temp.size(), final_n, layers);
				temp.resize(final_n);
				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<_Float16>, gpu::bitonic::RunStats>{
					elapsed_wall_ms, std::move(temp), stats};
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
#endif

// ==========================================
// MapReduce Hooks
// ==========================================

template <typename T> class GpuMapReduceHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
	explicit GpuMapReduceHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::MapReduceRunStats* stats) override {
		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats> {
				gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};
				auto t0 = std::chrono::high_resolution_clock::now();

				std::vector<T> output = gpu::map_reduce::run_topk(input.data(), input.size(), cfg.k, cfg.want_max,
																  cfg.ex_threads, &map_stats);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, gpu::map_reduce::RunStats>{
					elapsed_wall_ms, std::move(output), map_stats};
			});

		if (stats != nullptr) {
			stats->end_to_end_ms = best.elapsed_ms;
			stats->algorithm_ms = best.stats.elapsed_ms;
			stats->tiles_used = best.stats.tiles_used;
			stats->aggregated_candidates = best.stats.aggregated_candidates;
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

#if defined(__FLT16_MANT_DIG__)
class GpuMapReduceFp16Hooks final : public common::topk::MapReduceRunnerHooks<_Float16> {
  public:
	explicit GpuMapReduceFp16Hooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<_Float16> run(const std::vector<_Float16>& input, const Config& cfg,
							  common::topk::MapReduceRunStats* stats) override {
		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<_Float16>, gpu::map_reduce::RunStats> {
				gpu::map_reduce::RunStats map_stats{0.0, 0, 0, 0};
				std::vector<_Float16> output(cfg.k);
				auto t0 = std::chrono::high_resolution_clock::now();

				std::size_t count = gpu::map_reduce::run_topk_fp16(input.data(), input.size(), cfg.k, cfg.want_max,
																   cfg.ex_threads, output.data(), &map_stats);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				output.resize(count);
				return common::benchmark::TimedValueWithStats<std::vector<_Float16>, gpu::map_reduce::RunStats>{
					elapsed_wall_ms, std::move(output), map_stats};
			});

		if (stats != nullptr) {
			stats->end_to_end_ms = best.elapsed_ms;
			stats->algorithm_ms = best.stats.elapsed_ms;
			stats->tiles_used = best.stats.tiles_used;
			stats->aggregated_candidates = best.stats.aggregated_candidates;
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
#endif

// ==========================================
// Ground Truth Hooks
// ==========================================

template <typename T> class GpuGroundTruthHooks final : public common::topk::GroundTruthRunnerHooks<T> {
  public:
	explicit GpuGroundTruthHooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<T> run(const std::vector<T>& input, const Config& cfg,
					   common::topk::GroundTruthRunStats* stats) override {
		const std::size_t k = std::min(cfg.k, input.size());

		auto best =
			common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, double> {
				std::vector<T> temp = input;
				auto t0 = std::chrono::high_resolution_clock::now();

				double algo_ms = gpu::ground_truth::run_topk(temp, k, cfg.want_max);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, double>{elapsed_wall_ms, std::move(temp),
																					  algo_ms};
			});

		if (stats != nullptr) {
			stats->end_to_end_ms = best.elapsed_ms;
			stats->algorithm_ms = best.stats;
		}

		return std::move(best.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::GroundTruthRunStats& stats) override {
	}

  private:
	std::string device_name;
};

#if defined(__FLT16_MANT_DIG__)
class GpuGroundTruthFp16Hooks final : public common::topk::GroundTruthRunnerHooks<_Float16> {
  public:
	explicit GpuGroundTruthFp16Hooks() : device_name(gpu::bitonic::query_device_name()) {
	}

	void print_configuration(const Config& cfg, std::size_t n) override {
		gpu::reporting::print_configuration(cfg, n, device_name);
	}

	std::vector<_Float16> run(const std::vector<_Float16>& input, const Config& cfg,
							  common::topk::GroundTruthRunStats* stats) override {
		const std::size_t k = std::min(cfg.k, input.size());
		std::vector<float> input_f = to_float(input);

		auto best = common::benchmark::run_benchmark(
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<float>, double> {
				std::vector<float> temp = input_f;
				auto t0 = std::chrono::high_resolution_clock::now();

				double algo_ms = gpu::ground_truth::run_topk(temp, k, cfg.want_max);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<float>, double>{elapsed_wall_ms,
																						  std::move(temp), algo_ms};
			});

		if (stats != nullptr) {
			stats->end_to_end_ms = best.elapsed_ms;
			stats->algorithm_ms = best.stats;
		}

		return to_half(best.value);
	}

	void print_debug_metrics(const Config& cfg, const common::topk::GroundTruthRunStats& stats) override {
	}

  private:
	std::string device_name;
};
#endif

// ==========================================
// Dispatch
// ==========================================

template <typename T> int topk_typed(const Config& cfg) {
	if (cfg.algorithm == Algorithm::MapReduce) {
		GpuMapReduceHooks<T> hooks;
		return common::topk::execute_map_reduce<T>(cfg, hooks);
	} else if (cfg.algorithm == Algorithm::GroundTruth) {
		GpuGroundTruthHooks<T> hooks;
		return common::topk::execute_ground_truth<T>(cfg, hooks);
	}
	GpuBitonicRunnerHooks<T> hooks;
	return common::topk::execute_bitonic<T>(cfg, hooks);
}

int topk_fp16_dispatch(const Config& cfg) {
#if defined(__FLT16_MANT_DIG__)
	if (cfg.algorithm == Algorithm::MapReduce) {
		GpuMapReduceFp16Hooks hooks;
		return common::topk::execute_map_reduce<_Float16>(cfg, hooks);
	} else if (cfg.algorithm == Algorithm::GroundTruth) {
		GpuGroundTruthFp16Hooks hooks;
		return common::topk::execute_ground_truth<_Float16>(cfg, hooks);
	}
	GpuBitonicFp16Hooks hooks;
	return common::topk::execute_bitonic<_Float16>(cfg, hooks);
#else
	(void)cfg;
	throw std::invalid_argument("dtype=fp16 is not supported by this compiler target");
#endif
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
