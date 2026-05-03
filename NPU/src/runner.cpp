#include "../include/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../include/algorithm.hpp"
#include "../include/reporting.hpp"
#include "common/runner.hpp"
#include "common/benchmark.hpp"

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
        std::vector<T> data_backup = data; // Bitonic is in-place, need a pristine backup
        
        // warmup
        common::benchmark::warmup(common::benchmark::kWarmupIters, [&]() {
            std::vector<T> temp = data_backup;
            npu::bitonic::run_network_npu(temp, layers, context.ex_threads);
        });

        // measurement
        auto best = common::benchmark::measure_best(common::benchmark::kMeasureIters, [&]()
            -> common::benchmark::TimedValueWithStats<std::vector<T>, npu::bitonic::RunStats> {
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
        data = std::move(best.value); // Keep the successfully sorted array
        
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

template <typename T> int topk_typed(const Config& cfg) {
    const Context ctx = build_context(cfg);
    NpuBitonicRunnerHooks<T> hooks(ctx);
    return common::topk::execute_bitonic<T>(cfg, hooks);
}

} // namespace

int execute(const common::config::Config& cfg) {
    if (cfg.algorithm != Algorithm::Bitonic) {
        throw std::invalid_argument("NPU backend currently supports only algo=bitonic");
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
