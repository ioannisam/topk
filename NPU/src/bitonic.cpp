#include "../include/algorithm.hpp"
#include "xrt_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace npu::bitonic {

namespace {

template <typename T>
RunStats run_network_offload_xrt(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
                                 const npu::utils::OffloadConfig& offload_cfg) {
    const std::size_t n = data.size();
    
    std::size_t padded_n = n;
    if (n % 1024 != 0) {
        padded_n = ((n / 1024) + 1) * 1024;
    }

    std::vector<T> padded_data(padded_n, std::numeric_limits<T>::max());
    std::memcpy(padded_data.data(), data.data(), n * sizeof(T));

    npu::utils::SharedXrtState& state = npu::utils::get_shared_xrt_state(offload_cfg);

    const std::size_t chunk_bytes = 1024 * sizeof(T);
    
    xrt::bo src_bo(state.dev, padded_n * sizeof(T), xrt::bo::flags::host_only, npu::utils::safe_group_id(state.kernel, 3));
    xrt::bo dst_bo(state.dev, padded_n * sizeof(T), xrt::bo::flags::host_only, npu::utils::safe_group_id(state.kernel, 4));

    std::size_t total_dispatches = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (std::size_t offset = 0; offset < padded_n; offset += 1024) {
        std::memcpy(src_bo.map<void*>(), padded_data.data() + offset, chunk_bytes);
        src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(state.kernel);
        run.set_arg(0, 3);          
        run.set_arg(1, state.instr_bo);   
        run.set_arg(2, static_cast<uint32_t>(state.instr_v.size()));
        run.set_arg(3, src_bo);     
        run.set_arg(4, dst_bo);     
        
        xrt::runlist rl(state.hwctx);
        rl.add(run);
        rl.execute();
        npu::utils::wait_for_runlist_or_throw(rl, npu::utils::read_wait_timeout_ms());
        total_dispatches++;

        dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(padded_data.data() + offset, dst_bo.map<void*>(), chunk_bytes);
        
        if ((offset / 1024) % 2 != 0) {
            std::reverse(padded_data.begin() + offset, padded_data.begin() + offset + 1024);
        }
    }

    for (std::size_t k = 2048; k <= padded_n; k *= 2) {
        for (std::size_t j = k / 2; j > 0; j /= 2) {
            for (std::size_t i = 0; i < padded_n; i++) {
                std::size_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    
                    if (ascending) {
                        if (padded_data[i] > padded_data[ixj]) {
                            std::swap(padded_data[i], padded_data[ixj]);
                        }
                    } else {
                        if (padded_data[i] < padded_data[ixj]) {
                            std::swap(padded_data[i], padded_data[ixj]);
                        }
                    }
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    std::memcpy(data.data(), padded_data.data(), n * sizeof(T));

    return RunStats{std::chrono::duration<double, std::milli>(t1 - t0).count(), total_dispatches, 0, 1, true};
}

} // namespace

std::string query_device_name() {
    xrt::device dev = npu::utils::open_device();
    return dev.get_info<xrt::info::device::name>();
}

std::string query_device_bdf() {
    xrt::device dev = npu::utils::open_device();
    return dev.get_info<xrt::info::device::bdf>();
}

bool is_offload_configured() {
    return npu::utils::load_offload_config().enabled;
}

template <typename T>
RunStats run_network_npu(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers) {
    (void)workers;
    if (data.empty()) return RunStats{0.0, 0, 0, 1, true};

    (void)npu::utils::open_device();
    const npu::utils::OffloadConfig offload_cfg = npu::utils::load_offload_config();
    if (!offload_cfg.enabled) {
        throw std::runtime_error("NPU offload is required for this backend. Set NPU_OFFLOAD_XCLBIN to a valid xclbin path.");
    }
    return run_network_offload_xrt(data, layers, offload_cfg);
}

template RunStats run_network_npu<std::int32_t>(std::vector<std::int32_t>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<std::uint32_t>(std::vector<std::uint32_t>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template RunStats run_network_npu<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template RunStats run_network_npu<_Float16>(std::vector<_Float16>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
#endif

} // namespace npu::bitonic
