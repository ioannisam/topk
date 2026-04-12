#include "../include/algorithm.hpp"

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

#if defined(__x86_64__) || defined(__i386__)
bool cpu_supports_avx512f() {
	static const bool has_avx512f = []() {
#if defined(__GNUC__) || defined(__clang__)
		__builtin_cpu_init();
		return __builtin_cpu_supports("avx512f");
#else
		return false;
#endif
	}();
	return has_avx512f;
}

bool cpu_supports_avx2() {
	static const bool has_avx2 = []() {
#if defined(__GNUC__) || defined(__clang__)
		__builtin_cpu_init();
		return __builtin_cpu_supports("avx2");
#else
		return false;
#endif
	}();
	return has_avx2;
}
#endif

template <typename T> struct SimdThreshold {
	static std::size_t block_width(bool, bool) {
		return 1;
	}

	template <bool WantMax> static bool any_greater(const T*, T, bool, bool) {
		(void)WantMax;
		return true;
	}
};

#if defined(__x86_64__) || defined(__i386__)
template <> struct SimdThreshold<std::int32_t> {
	static std::size_t block_width(bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return 16;
		}
		if (use_avx2) {
			return 8;
		}
		return 1;
	}

	template <bool WantMax>
	__attribute__((target("avx512f"))) static bool any_greater_avx512(const std::int32_t* ptr, std::int32_t threshold) {
		const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
		const __m512i t = _mm512_set1_epi32(threshold);
		const __mmask16 mask = WantMax ? _mm512_cmpgt_epi32_mask(v, t) : _mm512_cmpgt_epi32_mask(t, v);
		return mask != 0;
	}

	template <bool WantMax>
	__attribute__((target("avx2"))) static bool any_greater_avx2(const std::int32_t* ptr, std::int32_t threshold) {
		const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
		const __m256i t = _mm256_set1_epi32(threshold);
		const __m256i cmp = WantMax ? _mm256_cmpgt_epi32(v, t) : _mm256_cmpgt_epi32(t, v);
		return _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) != 0;
	}

	template <bool WantMax>
	static bool any_greater(const std::int32_t* ptr, std::int32_t threshold, bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return any_greater_avx512<WantMax>(ptr, threshold);
		}
		if (use_avx2) {
			return any_greater_avx2<WantMax>(ptr, threshold);
		}
		return true;
	}
};

template <> struct SimdThreshold<std::uint32_t> {
	static std::size_t block_width(bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return 16;
		}
		if (use_avx2) {
			return 8;
		}
		return 1;
	}

	template <bool WantMax>
	__attribute__((target("avx512f"))) static bool any_greater_avx512(const std::uint32_t* ptr,
																	  std::uint32_t threshold) {
		const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
		const __m512i t = _mm512_set1_epi32(static_cast<std::int32_t>(threshold));
		const __mmask16 mask =
			WantMax ? _mm512_cmp_epu32_mask(v, t, _MM_CMPINT_GT) : _mm512_cmp_epu32_mask(t, v, _MM_CMPINT_GT);
		return mask != 0;
	}

	template <bool WantMax>
	__attribute__((target("avx2"))) static bool any_greater_avx2(const std::uint32_t* ptr, std::uint32_t threshold) {
		const __m256i sign = _mm256_set1_epi32(static_cast<std::int32_t>(0x80000000u));
		const __m256i v = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)), sign);
		const __m256i t = _mm256_xor_si256(_mm256_set1_epi32(static_cast<std::int32_t>(threshold)), sign);
		const __m256i cmp = WantMax ? _mm256_cmpgt_epi32(v, t) : _mm256_cmpgt_epi32(t, v);
		return _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) != 0;
	}

	template <bool WantMax>
	static bool any_greater(const std::uint32_t* ptr, std::uint32_t threshold, bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return any_greater_avx512<WantMax>(ptr, threshold);
		}
		if (use_avx2) {
			return any_greater_avx2<WantMax>(ptr, threshold);
		}
		return true;
	}
};

template <> struct SimdThreshold<float> {
	static std::size_t block_width(bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return 16;
		}
		if (use_avx2) {
			return 8;
		}
		return 1;
	}

	template <bool WantMax>
	__attribute__((target("avx512f"))) static bool any_greater_avx512(const float* ptr, float threshold) {
		const __m512 v = _mm512_loadu_ps(ptr);
		const __m512 t = _mm512_set1_ps(threshold);
		const __mmask16 mask = WantMax ? _mm512_cmp_ps_mask(v, t, _CMP_GT_OQ) : _mm512_cmp_ps_mask(v, t, _CMP_LT_OQ);
		return mask != 0;
	}

	template <bool WantMax>
	__attribute__((target("avx2"))) static bool any_greater_avx2(const float* ptr, float threshold) {
		const __m256 v = _mm256_loadu_ps(ptr);
		const __m256 t = _mm256_set1_ps(threshold);
		const __m256 cmp = WantMax ? _mm256_cmp_ps(v, t, _CMP_GT_OQ) : _mm256_cmp_ps(v, t, _CMP_LT_OQ);
		return _mm256_movemask_ps(cmp) != 0;
	}

	template <bool WantMax>
	static bool any_greater(const float* ptr, float threshold, bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return any_greater_avx512<WantMax>(ptr, threshold);
		}
		if (use_avx2) {
			return any_greater_avx2<WantMax>(ptr, threshold);
		}
		return true;
	}
};

template <> struct SimdThreshold<double> {
	static std::size_t block_width(bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return 8;
		}
		if (use_avx2) {
			return 4;
		}
		return 1;
	}

	template <bool WantMax>
	__attribute__((target("avx512f"))) static bool any_greater_avx512(const double* ptr, double threshold) {
		const __m512d v = _mm512_loadu_pd(ptr);
		const __m512d t = _mm512_set1_pd(threshold);
		const __mmask8 mask = WantMax ? _mm512_cmp_pd_mask(v, t, _CMP_GT_OQ) : _mm512_cmp_pd_mask(v, t, _CMP_LT_OQ);
		return mask != 0;
	}

	template <bool WantMax>
	__attribute__((target("avx2"))) static bool any_greater_avx2(const double* ptr, double threshold) {
		const __m256d v = _mm256_loadu_pd(ptr);
		const __m256d t = _mm256_set1_pd(threshold);
		const __m256d cmp = WantMax ? _mm256_cmp_pd(v, t, _CMP_GT_OQ) : _mm256_cmp_pd(v, t, _CMP_LT_OQ);
		return _mm256_movemask_pd(cmp) != 0;
	}

	template <bool WantMax>
	static bool any_greater(const double* ptr, double threshold, bool use_avx512f, bool use_avx2) {
		if (use_avx512f) {
			return any_greater_avx512<WantMax>(ptr, threshold);
		}
		if (use_avx2) {
			return any_greater_avx2<WantMax>(ptr, threshold);
		}
		return true;
	}
};
#endif

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
std::vector<T> map(const std::vector<T>& data, std::size_t begin, std::size_t end, std::size_t k, bool use_avx512f,
				   bool use_avx2) {

	using HeapCompare = std::conditional_t<WantMax, std::greater<T>, std::less<T>>;
	std::vector<T> heap;
	heap.reserve(k);
	if (k == 0 || begin >= end) {
		return {};
	}

	const std::size_t block = SimdThreshold<T>::block_width(use_avx512f, use_avx2);
	std::size_t i = begin;
	while (i < end) {
		if (heap.size() < k) {
			heap.push_back(data[i]);
			if (heap.size() == k) {
				// cheaper than pushing k times
				std::make_heap(heap.begin(), heap.end(), HeapCompare{});
			}
			++i;
			continue;
		}

		const T threshold = heap.front();
		const std::size_t remaining = end - i;
		if (block > 1 && remaining >= block &&
			!SimdThreshold<T>::template any_greater<WantMax>(data.data() + i, threshold, use_avx512f, use_avx2)) {
			i += block;
			continue;
		}

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

#if defined(__x86_64__) || defined(__i386__)
	const bool has_avx512f = cpu_supports_avx512f();
	const bool has_avx2 = cpu_supports_avx2();
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
