#include "../include/algorithm.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>

namespace npu::ground_truth {

template <typename T> void run_topk(std::vector<T>& data, std::size_t k, bool want_max) {
	if (k >= data.size()) {
		if (want_max) {
			std::sort(data.begin(), data.end(), std::greater<T>());
		} else {
			std::sort(data.begin(), data.end(), std::less<T>());
		}
	} else if (k > 0) {
		if (want_max) {
			std::partial_sort(data.begin(), data.begin() + k, data.end(), std::greater<T>());
		} else {
			std::partial_sort(data.begin(), data.begin() + k, data.end(), std::less<T>());
		}
	}
}

template void run_topk<std::int32_t>(std::vector<std::int32_t>& data, std::size_t k, bool want_max);
template void run_topk<std::uint32_t>(std::vector<std::uint32_t>& data, std::size_t k, bool want_max);
template void run_topk<float>(std::vector<float>& data, std::size_t k, bool want_max);
template void run_topk<double>(std::vector<double>& data, std::size_t k, bool want_max);

#if defined(__FLT16_MANT_DIG__)
template void run_topk<_Float16>(std::vector<_Float16>& data, std::size_t k, bool want_max);
#endif

} // namespace npu::ground_truth
