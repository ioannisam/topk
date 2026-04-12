#include "../include/algorithm.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace cpu::map_reduce {

namespace {

template <bool WantMax, typename T> bool scalar_is_candidate(T value, T threshold) {
	if constexpr (WantMax) {
		return value > threshold;
	}
	return value < threshold;
}

template <typename T, typename Compare> void sift_down(std::vector<T>& heap, std::size_t i, Compare comp) {
	const std::size_t n = heap.size();
	while (true) {
		std::size_t best = i;
		std::size_t left = 2 * i + 1;
		std::size_t right = 2 * i + 2;

		if (left < n && comp(heap[left], heap[best]))
			best = left;
		if (right < n && comp(heap[right], heap[best]))
			best = right;

		if (best != i) {
			std::swap(heap[i], heap[best]);
			i = best;
		} else {
			break;
		}
	}
}

template <bool WantMax, typename T>
std::vector<T> map(const std::vector<T>& data, std::size_t begin, std::size_t end, std::size_t k) {

	using HeapCompare = std::conditional_t<WantMax, std::greater<T>, std::less<T>>;
	std::vector<T> heap;
	heap.reserve(k);
	if (k == 0 || begin >= end) {
		return {};
	}

	for (std::size_t i = begin; i < end; ++i) {
		if (heap.size() < k) {
			heap.push_back(data[i]);
			if (heap.size() == k) {
				// cheaper than pushing k times
				std::make_heap(heap.begin(), heap.end(), HeapCompare{});
			}
			continue;
		}

		const T threshold = heap.front();
		if (!scalar_is_candidate<WantMax>(data[i], threshold)) {
			continue;
		}

		// avoid pop and push by replacing the root and sifting down
		heap[0] = data[i];
		sift_down(heap, 0, HeapCompare{});
	}

	return heap;
}

template <bool WantMax, typename T> std::vector<T> reduce(std::vector<T> aggregated, std::size_t k) {
	if (k == 0 || aggregated.empty()) {
		return {};
	}

	auto cmp = [](const T& lhs, const T& rhs) {
		if constexpr (WantMax) {
			return lhs > rhs;
		}
		return lhs < rhs;
	};

	if (aggregated.size() > k) {
		std::nth_element(aggregated.begin(), aggregated.begin() + static_cast<std::ptrdiff_t>(k), aggregated.end(),
						 cmp);
		aggregated.resize(k);
	}

	std::sort(aggregated.begin(), aggregated.end(), cmp);
	return aggregated;
}

template <bool WantMax, typename T>
std::vector<T> topk(const std::vector<T>& data, std::size_t k, std::size_t workers, RunStats* stats) {
	const std::size_t n = data.size();
	k = std::min(k, n);
	workers = std::max<std::size_t>(1, std::min(workers, n));

	std::vector<std::vector<T>> local_topk(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers > 0 ? workers - 1 : 0);

	for (std::size_t tid = 0; tid + 1 < workers; ++tid) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;
			local_topk[tid] = map<WantMax>(data, begin, end, k);
		});
	}
	// avoid main thread becoming idle
	const std::size_t last_tid = workers - 1;
	const std::size_t begin = (n * last_tid) / workers;
	const std::size_t end = n;
	local_topk[last_tid] = map<WantMax>(data, begin, end, k);

	for (auto& t : pool) {
		t.join();
	}

	std::vector<T> aggregated;
	aggregated.reserve(k * workers);
	for (const auto& tile_values : local_topk) {
		aggregated.insert(aggregated.end(), tile_values.begin(), tile_values.end());
	}

	if (stats != nullptr) {
		stats->tiles_used = workers;
		stats->aggregated_candidates = aggregated.size();
	}

	return reduce<WantMax>(std::move(aggregated), k);
}

} // namespace

template <typename T>
std::vector<T> run_topk(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers,
						RunStats* stats) {
	if (want_max) {
		return topk<true>(data, k, workers, stats);
	}
	return topk<false>(data, k, workers, stats);
}

template std::vector<std::int32_t> run_topk<std::int32_t>(const std::vector<std::int32_t>& data, std::size_t k,
														  bool want_max, std::size_t workers, RunStats* stats);
template std::vector<std::uint32_t> run_topk<std::uint32_t>(const std::vector<std::uint32_t>& data, std::size_t k,
															bool want_max, std::size_t workers, RunStats* stats);
template std::vector<float> run_topk<float>(const std::vector<float>& data, std::size_t k, bool want_max,
											std::size_t workers, RunStats* stats);
template std::vector<double> run_topk<double>(const std::vector<double>& data, std::size_t k, bool want_max,
											  std::size_t workers, RunStats* stats);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> run_topk<_Float16>(const std::vector<_Float16>& data, std::size_t k, bool want_max,
												  std::size_t workers, RunStats* stats);
#endif

} // namespace cpu::map_reduce
