#include "../include/algorithm.hpp"

#include <algorithm>
#include <cstdint>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

namespace gpu::ground_truth {

template <typename T>
void run_topk(std::vector<T>& data, std::size_t k, bool want_max) {
    if (k == 0) {
        data.clear();
        return;
    }

    thrust::device_vector<T> d_vec = data;

    if (want_max) {
        thrust::sort(thrust::device, d_vec.begin(), d_vec.end(), thrust::greater<T>());
    } else {
        thrust::sort(thrust::device, d_vec.begin(), d_vec.end(), thrust::less<T>());
    }

    std::size_t out_size = std::min(k, data.size());
    data.resize(out_size);
    thrust::copy_n(d_vec.begin(), out_size, data.begin());
}

// Explicit instantiations
template void run_topk<std::int32_t>(std::vector<std::int32_t>& data, std::size_t k, bool want_max);
template void run_topk<std::uint32_t>(std::vector<std::uint32_t>& data, std::size_t k, bool want_max);
template void run_topk<float>(std::vector<float>& data, std::size_t k, bool want_max);
template void run_topk<double>(std::vector<double>& data, std::size_t k, bool want_max);

} // namespace gpu::ground_truth
