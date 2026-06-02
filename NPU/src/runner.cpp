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
#include "common/random.hpp"
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

		// warmup
		common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
			std::vector<T> temp = data_backup;
			npu::bitonic::run_network_npu(temp, layers, context.ex_threads);
		});

		// measurement
		auto best = common::benchmark::measure_best(
			common::benchmark::kMeasureIters,
			[&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats> {
				std::vector<T> temp = data_backup;

				auto t0 = std::chrono::high_resolution_clock::now();

				const npu::bitonic::RunStats stats = npu::bitonic::run_network_npu(temp, layers, context.ex_threads);

				auto t1 = std::chrono::high_resolution_clock::now();
				double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

				return common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats>{
					elapsed_wall_ms,
					std::move(temp),
					stats,
				};
			});

		npu::bitonic::RunStats best_stats = best.stats;
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

	void print_debug_metrics(const Config& cfg, std::size_t layer_count, std::size_t full_cmp, std::size_t trunc_cmp,
							 const common::topk::BasicRunStats* full_stats,
							 const common::topk::BasicRunStats* trunc_stats) override {
		const npu::bitonic::RunStats* full_run = (full_stats != nullptr) ? &last_full_stats : nullptr;
		const npu::bitonic::RunStats* trunc_run = (trunc_stats != nullptr) ? &last_trunc_stats : nullptr;
		npu::reporting::print_debug_metrics(cfg, context.device_name, full_run, trunc_run, layer_count, full_cmp,
											trunc_cmp);
	}

  private:
	Context context;
	npu::bitonic::RunStats last_full_stats{0.0, 0, 0, 0, false};
	npu::bitonic::RunStats last_trunc_stats{0.0, 0, 0, 0, false};
};

template <typename T>
class NpuMapReduceRunnerHooks final : public common::topk::MapReduceRunnerHooks<T> {
  public:
    explicit NpuMapReduceRunnerHooks(const Context& ctx) : context(ctx) {
    }

    void print_configuration(const Config& cfg, std::size_t n) override {
        npu::reporting::print_configuration(cfg, n, context.ex_threads, context.device_name, context.device_bdf,
                                            context.offload_enabled);
    }

    std::vector<T> run(const std::vector<T>& input, const Config& cfg,
                       common::topk::MapReduceRunStats* stats) override {
        // warmup
        common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
            npu::bitonic::RunStats dummy_stats{0.0, 0, 0, 0, false};
            npu::map_reduce::run_topk_npu(input, cfg.k, cfg.want_max, context.ex_threads, &dummy_stats);
        });

        // measurement
        auto best = common::benchmark::measure_best(
            common::benchmark::kMeasureIters,
            [&]() -> common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats> {
                npu::bitonic::RunStats run_stats{0.0, 0, 0, 0, false};
                auto t0 = std::chrono::high_resolution_clock::now();

                std::vector<T> output =
                    npu::map_reduce::run_topk_npu(input, cfg.k, cfg.want_max, context.ex_threads, &run_stats);

                auto t1 = std::chrono::high_resolution_clock::now();
                double elapsed_wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

                return common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats>{
                    elapsed_wall_ms,
                    std::move(output),
                    run_stats,
                };
            });

        last_run_stats = best.stats;
        if (stats != nullptr) {
            stats->end_to_end_ms = best.elapsed_ms;
            stats->algorithm_ms = best.stats.elapsed_ms;
            stats->tiles_used = 0;
            stats->aggregated_candidates = 0;
            std::cout << "[PROFILE_TIME_MS] " << stats->end_to_end_ms << "\n";
        } else {
            std::cout << "[PROFILE_TIME_MS] " << best.elapsed_ms << "\n";
        }

        return std::move(best.value);
    }

    void print_debug_metrics(const Config& cfg, const common::topk::MapReduceRunStats& stats) override {
        if (!cfg.debug_output) {
            return;
        }

        common::reporting::print_section_header("Debug Metrics");
        common::reporting::print_key_value("NPU device", context.device_name);
        common::reporting::print_key_value("NPU BDF", context.device_bdf);
        common::reporting::print_key_value("Offload configured", (context.offload_enabled ? "yes" : "no"));
        common::reporting::print_key_value("Tiles used", stats.tiles_used);
        common::reporting::print_key_value("Aggregated candidates", stats.aggregated_candidates);
        common::reporting::print_key_value("NPU dispatches", last_run_stats.layer_dispatches);
        common::reporting::print_key_value("Active comparators", last_run_stats.active_comparators);
        common::reporting::print_key_value("Execution workers", last_run_stats.workers);
    }

  private:
    Context context;
    npu::bitonic::RunStats last_run_stats{0.0, 0, 0, 0, false};
};

template <typename T> int topk_typed(const Config& cfg) {
    const Context ctx = build_context(cfg);

    if (cfg.algorithm == Algorithm::Bitonic) {
        NpuBitonicRunnerHooks<T> hooks(ctx);
        return common::topk::execute_bitonic<T>(cfg, hooks);
    }
    NpuMapReduceRunnerHooks<T> hooks(ctx);
    return common::topk::execute_map_reduce<T>(cfg, hooks);
}

} // namespace

int execute(const common::config::Config& cfg) {
    if (cfg.algorithm != Algorithm::Bitonic && cfg.algorithm != Algorithm::MapReduce) {
        throw std::invalid_argument("NPU backend currently supports only algo=bitonic or algo=map_reduce");
    }

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
