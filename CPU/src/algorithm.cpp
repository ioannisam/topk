#include "../include/algorithm.hpp"

#include <condition_variable>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>

namespace cpu::bitonic {

namespace {

#if defined(__x86_64__) || defined(__i386__)
bool cpu_supports_avx512f() {
	static const bool has_avx512 = []() {
#if defined(__GNUC__) || defined(__clang__)
		__builtin_cpu_init();
		return __builtin_cpu_supports("avx512f");
#else
		return false;
#endif
	}();
	return has_avx512;
}

__attribute__((target("avx512f"))) void run_layer_j1_avx512_i32(std::int32_t* ptr, std::size_t begin, std::size_t end,
											  std::size_t k) {
	const __m512i swap_idx = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
	const __m512i lane_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const __m512i one = _mm512_set1_epi32(1);
	const __mmask16 odd_mask = 0xAAAA;
	const __m512i kvec = _mm512_set1_epi32(static_cast<std::int32_t>(k));

	std::size_t i = begin + (begin & 1U);
	for (; i + 15 < end; i += 16) {
		const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr + i));
		const __m512i swapped = _mm512_permutexvar_epi32(swap_idx, v);
		const __m512i lo = _mm512_min_epi32(v, swapped);
		const __m512i hi = _mm512_max_epi32(v, swapped);

		const __m512i base = _mm512_add_epi32(_mm512_set1_epi32(static_cast<std::int32_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(one, base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, kvec);
		const __mmask16 desc_mask = _mm512_cmpneq_epi32_mask(dir_bits, _mm512_setzero_si512());
		const __mmask16 choose_hi = odd_mask ^ desc_mask;

		const __m512i out = _mm512_mask_blend_epi32(choose_hi, lo, hi);
		_mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr + i), out);
	}

	for (; i < end; ++i) {
		const std::size_t ixj = i ^ std::size_t{1};
		if (ixj <= i) {
			continue;
		}
		const bool ascending = (i & k) == 0;
		if (ascending) {
			if (ptr[i] > ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		} else {
			if (ptr[i] < ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		}
	}
}

__attribute__((target("avx512f"))) void run_layer_j1_avx512_u32(std::uint32_t* ptr, std::size_t begin,
											   std::size_t end, std::size_t k) {
	const __m512i swap_idx = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
	const __m512i lane_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const __m512i one = _mm512_set1_epi32(1);
	const __mmask16 odd_mask = 0xAAAA;
	const __m512i kvec = _mm512_set1_epi32(static_cast<std::int32_t>(k));

	std::size_t i = begin + (begin & 1U);
	for (; i + 15 < end; i += 16) {
		const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr + i));
		const __m512i swapped = _mm512_permutexvar_epi32(swap_idx, v);
		const __m512i lo = _mm512_min_epu32(v, swapped);
		const __m512i hi = _mm512_max_epu32(v, swapped);

		const __m512i base = _mm512_add_epi32(_mm512_set1_epi32(static_cast<std::int32_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(one, base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, kvec);
		const __mmask16 desc_mask = _mm512_cmpneq_epi32_mask(dir_bits, _mm512_setzero_si512());
		const __mmask16 choose_hi = odd_mask ^ desc_mask;

		const __m512i out = _mm512_mask_blend_epi32(choose_hi, lo, hi);
		_mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr + i), out);
	}

	for (; i < end; ++i) {
		const std::size_t ixj = i ^ std::size_t{1};
		if (ixj <= i) {
			continue;
		}
		const bool ascending = (i & k) == 0;
		if (ascending) {
			if (ptr[i] > ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		} else {
			if (ptr[i] < ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		}
	}
}

__attribute__((target("avx512f"))) void run_layer_j1_avx512_f32(float* ptr, std::size_t begin, std::size_t end,
											  std::size_t k) {
	const __m512i swap_idx = _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
	const __m512i lane_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	const __m512i one = _mm512_set1_epi32(1);
	const __mmask16 odd_mask = 0xAAAA;
	const __m512i kvec = _mm512_set1_epi32(static_cast<std::int32_t>(k));

	std::size_t i = begin + (begin & 1U);
	for (; i + 15 < end; i += 16) {
		const __m512 v = _mm512_loadu_ps(ptr + i);
		const __m512 swapped = _mm512_permutexvar_ps(swap_idx, v);
		const __m512 lo = _mm512_min_ps(v, swapped);
		const __m512 hi = _mm512_max_ps(v, swapped);

		const __m512i base = _mm512_add_epi32(_mm512_set1_epi32(static_cast<std::int32_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(one, base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, kvec);
		const __mmask16 desc_mask = _mm512_cmpneq_epi32_mask(dir_bits, _mm512_setzero_si512());
		const __mmask16 choose_hi = odd_mask ^ desc_mask;

		const __m512 out = _mm512_mask_blend_ps(choose_hi, lo, hi);
		_mm512_storeu_ps(ptr + i, out);
	}

	for (; i < end; ++i) {
		const std::size_t ixj = i ^ std::size_t{1};
		if (ixj <= i) {
			continue;
		}
		const bool ascending = (i & k) == 0;
		if (ascending) {
			if (ptr[i] > ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		} else {
			if (ptr[i] < ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		}
	}
}

__attribute__((target("avx512f"))) void run_layer_j1_avx512_f64(double* ptr, std::size_t begin, std::size_t end,
											  std::size_t k) {
	const __m512i swap_idx = _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6);
	const __m512i lane_idx = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
	const __m512i one = _mm512_set1_epi64(1);
	const __mmask8 odd_mask = 0xAA;
	const __m512i kvec = _mm512_set1_epi64(static_cast<std::int64_t>(k));

	std::size_t i = begin + (begin & 1U);
	for (; i + 7 < end; i += 8) {
		const __m512d v = _mm512_loadu_pd(ptr + i);
		const __m512d swapped = _mm512_permutexvar_pd(swap_idx, v);
		const __m512d lo = _mm512_min_pd(v, swapped);
		const __m512d hi = _mm512_max_pd(v, swapped);

		const __m512i base = _mm512_add_epi64(_mm512_set1_epi64(static_cast<std::int64_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(one, base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, kvec);
		const __mmask8 desc_mask = _mm512_cmpneq_epi64_mask(dir_bits, _mm512_setzero_si512());
		const __mmask8 choose_hi = odd_mask ^ desc_mask;

		const __m512d out = _mm512_mask_blend_pd(choose_hi, lo, hi);
		_mm512_storeu_pd(ptr + i, out);
	}

	for (; i < end; ++i) {
		const std::size_t ixj = i ^ std::size_t{1};
		if (ixj <= i) {
			continue;
		}
		const bool ascending = (i & k) == 0;
		if (ascending) {
			if (ptr[i] > ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		} else {
			if (ptr[i] < ptr[ixj]) {
				std::swap(ptr[i], ptr[ixj]);
			}
		}
	}
}

template <typename T>
bool try_run_avx512_j1_layer(std::vector<T>& data, std::size_t begin, std::size_t end, std::size_t k, std::size_t j,
							 bool trunc) {
	if (trunc || j != 1 || !cpu_supports_avx512f()) {
		return false;
	}

	if (k > std::numeric_limits<std::int32_t>::max() || end > std::numeric_limits<std::int32_t>::max()) {
		return false;
	}

	if constexpr (std::is_same_v<T, std::int32_t>) {
		run_layer_j1_avx512_i32(data.data(), begin, end, k);
		return true;
	} else if constexpr (std::is_same_v<T, std::uint32_t>) {
		run_layer_j1_avx512_u32(data.data(), begin, end, k);
		return true;
	} else if constexpr (std::is_same_v<T, float>) {
		run_layer_j1_avx512_f32(data.data(), begin, end, k);
		return true;
	} else if constexpr (std::is_same_v<T, double>) {
		run_layer_j1_avx512_f64(data.data(), begin, end, k);
		return true;
	}

	return false;
}
#else
template <typename T>
bool try_run_avx512_j1_layer(std::vector<T>&, std::size_t, std::size_t, std::size_t, std::size_t, bool) {
	return false;
}
#endif

class Barrier {

  public:
	explicit Barrier(std::size_t participants) : threshold(participants), count(participants), generation(0) {
	}

	void wait() {
		std::unique_lock<std::mutex> lock(mutex);
		const std::size_t gen = generation;
		if (--count == 0) {
			generation++;
			count = threshold;
			cv.notify_all();
			return;
		}
		cv.wait(lock, [&] { return generation != gen; });
	}

  private:
	std::mutex mutex;
	std::condition_variable cv;
	std::size_t threshold;
	std::size_t count;
	std::size_t generation;
};

} // namespace

template <typename T>
void run_network_parallel(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers) {

	const std::size_t n = data.size();
	Barrier barrier(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers);

	for (std::size_t tid = 0; tid < workers; tid++) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;

			for (std::size_t layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
				const std::size_t k = layers[layer_idx].k;
				const std::size_t j = layers[layer_idx].j;

				if (try_run_avx512_j1_layer(data, begin, end, k, j, trunc)) {
					barrier.wait();
					continue;
				}

				for (std::size_t i = begin; i < end; ++i) {
					const std::size_t ixj = i ^ j;
					if (ixj <= i || ixj >= n) {
						continue;
					}

					if (trunc && !(keep[layer_idx][i] || keep[layer_idx][ixj])) {
						continue;
					}

					const bool ascending = (i & k) == 0;
					if (ascending) {
						if (data[i] > data[ixj]) {
							std::swap(data[i], data[ixj]);
						}
					} else {
						if (data[i] < data[ixj]) {
							std::swap(data[i], data[ixj]);
						}
					}
				}

				barrier.wait();
			}
		});
	}

	for (auto& t : pool) {
		t.join();
	}
}

template void run_network_parallel<std::int32_t>(std::vector<std::int32_t>& data, const std::vector<common::bitonic::Layer>& layers,
												 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
												 std::size_t workers);
template void run_network_parallel<std::uint32_t>(std::vector<std::uint32_t>& data, const std::vector<common::bitonic::Layer>& layers,
												  const std::vector<std::vector<unsigned char>>& keep, bool trunc,
												  std::size_t workers);
template void run_network_parallel<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
										  const std::vector<std::vector<unsigned char>>& keep, bool trunc,
										  std::size_t workers);
template void run_network_parallel<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers,
										   const std::vector<std::vector<unsigned char>>& keep, bool trunc,
										   std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template void run_network_parallel<_Float16>(std::vector<_Float16>& data, const std::vector<common::bitonic::Layer>& layers,
											 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
											 std::size_t workers);
#endif

} // namespace cpu::bitonic
