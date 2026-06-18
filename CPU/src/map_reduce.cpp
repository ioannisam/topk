#include "../include/algorithm.hpp"
#include "../include/simd_traits.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <immintrin.h>
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

		if (left < n && comp(heap[best], heap[left]))
			best = left;
		if (right < n && comp(heap[best], heap[right]))
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
std::vector<T> map(const std::vector<T>& data, std::size_t begin, std::size_t end, std::size_t k, bool use_avx512f,
				   bool use_avx2) {

	using HeapCompare = std::conditional_t<WantMax, std::greater<T>, std::less<T>>;
	std::vector<T> heap;
	heap.reserve(k);
	if (k == 0 || begin >= end) {
		return {};
	}

	// bulk initialization
	std::size_t initial_elements = std::min(k, end - begin);
	heap.assign(data.begin() + begin, data.begin() + begin + initial_elements);
	if (!heap.empty()) {
		std::make_heap(heap.begin(), heap.end(), HeapCompare{});
	}

	const std::size_t block = cpu::simd::simd_block_width<T>(use_avx512f, use_avx2);
	std::size_t i = begin + initial_elements;

	// hot loop!
	while (i < end) {
		const T threshold = heap.front();
		const std::size_t remaining = end - i;

		if (block > 1) {
            auto process_mask = [&](std::uint64_t mask, std::size_t offset) {
                while (mask != 0) {
                    int bit_idx = std::countr_zero(mask);
                    T candidate = data[offset + bit_idx];

                    if (scalar_is_candidate<WantMax>(candidate, heap.front())) {
                        heap[0] = candidate;
                        sift_down(heap, 0, HeapCompare{});
                    }
                    mask &= (mask - 1);
                }
            };

			if (remaining >= block * 4) {
				std::uint64_t m0 = cpu::simd::get_candidate_mask_simd<WantMax, T>(data.data() + i, threshold, use_avx512f, use_avx2);
				std::uint64_t m1 = cpu::simd::get_candidate_mask_simd<WantMax, T>(data.data() + i + block, threshold, use_avx512f, use_avx2);
				std::uint64_t m2 = cpu::simd::get_candidate_mask_simd<WantMax, T>(data.data() + i + block * 2, threshold, use_avx512f, use_avx2);
				std::uint64_t m3 = cpu::simd::get_candidate_mask_simd<WantMax, T>(data.data() + i + block * 3, threshold, use_avx512f, use_avx2);

				if ((m0 | m1 | m2 | m3) == 0) {
					i += block * 4;
					continue;
				}

                process_mask(m0, i);
                process_mask(m1, i + block);
                process_mask(m2, i + block * 2);
                process_mask(m3, i + block * 3);
                i += block * 4;
                continue;
			}

			// fallback for single blocks
			if (remaining >= block) {
				std::uint64_t mask = cpu::simd::get_candidate_mask_simd<WantMax, T>(data.data() + i, threshold, use_avx512f, use_avx2);
				process_mask(mask, i);
				i += block;
				continue;
			}
		}

		// tail elements where remaining < block
		if (!scalar_is_candidate<WantMax>(data[i], threshold)) {
			++i;
			continue;
		}

		// avoid pop and push by replacing the root and sifting down
		heap[0] = data[i];
		sift_down(heap, 0, HeapCompare{});
		++i;
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

	// capped at size <= k and is a valid heap
	std::sort(aggregated.begin(), aggregated.end(), cmp);
	return aggregated;
}

template <bool WantMax, typename T>
std::vector<T> topk(const std::vector<T>& data, std::size_t k, std::size_t workers, RunStats* stats) {
	const std::size_t n = data.size();
	k = std::min(k, n);
	workers = std::max<std::size_t>(1, std::min(workers, n));
	workers = std::min<std::size_t>(workers, 8);
	workers = std::min(workers, std::max<std::size_t>(1, n >> 20));

#if defined(__x86_64__) || defined(__i386__)
	const bool has_avx512f = cpu::simd::cpu_supports_avx512f();
	const bool has_avx2 = cpu::simd::cpu_supports_avx2();
	const char* force_avx512 = std::getenv("TOPK_FORCE_AVX512");
	const bool use_avx512f = has_avx512f && force_avx512 != nullptr && force_avx512[0] == '1';
	const bool use_avx2 = has_avx2 && !use_avx512f;
#else
	const bool use_avx512f = false;
	const bool use_avx2 = false;
#endif

	std::vector<std::vector<T>> local_topk(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers > 0 ? workers - 1 : 0);

	for (std::size_t tid = 0; tid + 1 < workers; ++tid) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;
			local_topk[tid] = map<WantMax>(data, begin, end, k, use_avx512f, use_avx2);
		});
	}
	// avoid main thread becoming idle
	const std::size_t last_tid = workers - 1;
	const std::size_t begin = (n * last_tid) / workers;
	const std::size_t end = n;
	local_topk[last_tid] = map<WantMax>(data, begin, end, k, use_avx512f, use_avx2);

	for (auto& t : pool) {
		t.join();
	}

	using HeapCompare = std::conditional_t<WantMax, std::greater<T>, std::less<T>>;
	std::vector<T>& final_heap = local_topk[0];

	for (std::size_t tid = 1; tid < local_topk.size(); ++tid) {
		for (const auto& candidate : local_topk[tid]) {
			if (final_heap.size() < k) {
				final_heap.push_back(candidate);
				if (final_heap.size() == k) {
					std::make_heap(final_heap.begin(), final_heap.end(), HeapCompare{});
				}
			} else if (scalar_is_candidate<WantMax>(candidate, final_heap.front())) {
				final_heap[0] = candidate;
				sift_down(final_heap, 0, HeapCompare{});
			}
		}
	}

	// total dataset across all threads < K
	if (!final_heap.empty() && final_heap.size() < k) {
		std::make_heap(final_heap.begin(), final_heap.end(), HeapCompare{});
	}

	if (stats != nullptr) {
		stats->tiles_used = workers;
		stats->aggregated_candidates = final_heap.size();
	}

	return reduce<WantMax>(std::move(final_heap), k);
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
