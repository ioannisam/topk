#include "../include/algorithm.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

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

template <typename T> struct SimdTraits;

// --- Float Traits ---
template <> struct SimdTraits<float> {
	using Vec = __m512;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	static Vec load(const float* p) { return _mm512_loadu_ps(p); }
	static void store(float* p, Vec v) { _mm512_storeu_ps(p, v); }
	static Vec min(Vec a, Vec b) { return _mm512_min_ps(a, b); }
	static Vec max(Vec a, Vec b) { return _mm512_max_ps(a, b); }
	static Vec blend(Mask m, Vec a, Vec b) { return _mm512_mask_blend_ps(m, a, b); }

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) return _mm512_permutexvar_ps(_mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14), v);
		if constexpr (J == 2) return _mm512_permutexvar_ps(_mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13), v);
		if constexpr (J == 4) return _mm512_permutexvar_ps(_mm512_setr_epi32(4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11), v);
		if constexpr (J == 8) return _mm512_permutexvar_ps(_mm512_setr_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7), v);
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		const __m512i lane_idx = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
		const __m512i base = _mm512_add_epi32(_mm512_set1_epi32(static_cast<std::int32_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(_mm512_set1_epi32(J), base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, _mm512_set1_epi32(static_cast<std::int32_t>(k)));
		const Mask desc_mask = _mm512_cmpneq_epi32_mask(dir_bits, _mm512_setzero_si512());
		
		Mask odd_mask = 0;
		if constexpr (J == 1) odd_mask = 0xAAAA;
		else if constexpr (J == 2) odd_mask = 0xCCCC;
		else if constexpr (J == 4) odd_mask = 0xF0F0;
		else if constexpr (J == 8) odd_mask = 0xFF00;
		return odd_mask ^ desc_mask;
	}
};

// --- Int32 Traits ---
template <> struct SimdTraits<std::int32_t> {
	using Vec = __m512i;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	static Vec load(const std::int32_t* p) { return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p)); }
	static void store(std::int32_t* p, Vec v) { _mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v); }
	static Vec min(Vec a, Vec b) { return _mm512_min_epi32(a, b); }
	static Vec max(Vec a, Vec b) { return _mm512_max_epi32(a, b); }
	static Vec blend(Mask m, Vec a, Vec b) { return _mm512_mask_blend_epi32(m, a, b); }

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) return _mm512_permutexvar_epi32(_mm512_setr_epi32(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14), v);
		if constexpr (J == 2) return _mm512_permutexvar_epi32(_mm512_setr_epi32(2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13), v);
		if constexpr (J == 4) return _mm512_permutexvar_epi32(_mm512_setr_epi32(4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11), v);
		if constexpr (J == 8) return _mm512_permutexvar_epi32(_mm512_setr_epi32(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7), v);
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		return SimdTraits<float>::template get_blend_mask<J>(i, k); 
	}
};

// --- UInt32 Traits ---
template <> struct SimdTraits<std::uint32_t> {
	using Vec = __m512i;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	static Vec load(const std::uint32_t* p) { return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p)); }
	static void store(std::uint32_t* p, Vec v) { _mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v); }
	static Vec min(Vec a, Vec b) { return _mm512_min_epu32(a, b); } 
	static Vec max(Vec a, Vec b) { return _mm512_max_epu32(a, b); }
	static Vec blend(Mask m, Vec a, Vec b) { return _mm512_mask_blend_epi32(m, a, b); }

	template <int J> static Vec permutex(Vec v) { return SimdTraits<std::int32_t>::template permutex<J>(v); }
	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) { return SimdTraits<float>::template get_blend_mask<J>(i, k); }
};

// --- Double Traits ---
template <> struct SimdTraits<double> {
	using Vec = __m512d;
	using Mask = __mmask8;
	static constexpr std::size_t width = 8;

	static Vec load(const double* p) { return _mm512_loadu_pd(p); }
	static void store(double* p, Vec v) { _mm512_storeu_pd(p, v); }
	static Vec min(Vec a, Vec b) { return _mm512_min_pd(a, b); }
	static Vec max(Vec a, Vec b) { return _mm512_max_pd(a, b); }
	static Vec blend(Mask m, Vec a, Vec b) { return _mm512_mask_blend_pd(m, a, b); }

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) return _mm512_permutexvar_pd(_mm512_setr_epi64(1,0,3,2,5,4,7,6), v);
		if constexpr (J == 2) return _mm512_permutexvar_pd(_mm512_setr_epi64(2,3,0,1,6,7,4,5), v);
		if constexpr (J == 4) return _mm512_permutexvar_pd(_mm512_setr_epi64(4,5,6,7,0,1,2,3), v);
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		const __m512i lane_idx = _mm512_setr_epi64(0,1,2,3,4,5,6,7);
		const __m512i base = _mm512_add_epi64(_mm512_set1_epi64(static_cast<std::int64_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(_mm512_set1_epi64(J), base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, _mm512_set1_epi64(static_cast<std::int64_t>(k)));
		const Mask desc_mask = _mm512_cmpneq_epi64_mask(dir_bits, _mm512_setzero_si512());
		
		Mask odd_mask = 0;
		if constexpr (J == 1) odd_mask = 0xAA;
		else if constexpr (J == 2) odd_mask = 0xCC;
		else if constexpr (J == 4) odd_mask = 0xF0;
		return odd_mask ^ desc_mask;
	}
};

// --- Unified Intra-Kernel (j < SIMD_WIDTH) ---
template <typename T, int J>
__attribute__((target("avx512f")))
void run_layer_intra_avx512(T* ptr, std::size_t begin, std::size_t end, std::size_t k) {
	using Traits = SimdTraits<T>;
	const std::size_t j2 = static_cast<std::size_t>(J) * 2;
	std::size_t i = (begin + (j2 - 1)) & ~(j2 - 1);
	
	for (; i + Traits::width - 1 < end; i += Traits::width) {
		auto v = Traits::load(ptr + i);
		auto swapped = Traits::template permutex<J>(v);
		auto lo = Traits::min(v, swapped);
		auto hi = Traits::max(v, swapped);
		
		auto mask = Traits::template get_blend_mask<J>(i, k);
		auto out = Traits::blend(mask, lo, hi);
		Traits::store(ptr + i, out);
	}

	// Scalar fallback for remaining elements
	for (; i < end; ++i) {
		const std::size_t ixj = i ^ static_cast<std::size_t>(J);
		if (ixj <= i) continue;
		const bool ascending = (i & k) == 0;
		if (ascending) { if (ptr[i] > ptr[ixj]) std::swap(ptr[i], ptr[ixj]); } 
		else { if (ptr[i] < ptr[ixj]) std::swap(ptr[i], ptr[ixj]); }
	}
}

// --- Unified Inter-Kernel (j >= SIMD_WIDTH) WITHOUT Unrolling ---
template <typename T>
__attribute__((target("avx512f")))
void run_layer_inter_avx512(T* ptr, std::size_t begin, std::size_t end, std::size_t k, std::size_t j, std::size_t n) {
	using Traits = SimdTraits<T>;
	std::size_t i = begin;
	
	while (i < end) {
		if ((i & j) != 0) {
			i = (i | ((j << 1) - 1)) + 1;
			continue;
		}

		std::size_t chunk_end = std::min((i | (j - 1)) + 1, end);
		if (chunk_end > n) chunk_end = n;

		for (; i + Traits::width - 1 < chunk_end; i += Traits::width) {
			const bool asc = (i & k) == 0;
			std::size_t ixj = i + j;

			auto v1 = Traits::load(ptr + i);
			auto v2 = Traits::load(ptr + ixj);

			auto lo = Traits::min(v1, v2);
			auto hi = Traits::max(v1, v2);

			if (asc) {
				Traits::store(ptr + i, lo);
				Traits::store(ptr + ixj, hi);
			} else {
				Traits::store(ptr + i, hi);
				Traits::store(ptr + ixj, lo);
			}
		}

		// Scalar fallback
		for (; i < chunk_end; ++i) {
			const std::size_t ixj = i + j;
			const bool ascending = (i & k) == 0;
			if (ascending) { if (ptr[i] > ptr[ixj]) std::swap(ptr[i], ptr[ixj]); } 
			else { if (ptr[i] < ptr[ixj]) std::swap(ptr[i], ptr[ixj]); }
		}
	}
}

// --- Dispatcher ---
template <typename T>
bool try_run_avx512_layer(std::vector<T>& data, std::size_t begin, std::size_t end, std::size_t k, std::size_t j, bool trunc) {
	if (!cpu_supports_avx512f()) return false;
	if (k > std::numeric_limits<std::int32_t>::max() || end > std::numeric_limits<std::int32_t>::max()) return false;

	const std::size_t n = data.size();
	T* ptr = data.data();

	// Strict constexpr branches prevent the compiler from instantiating AVX templates for unsupported types (like _Float16)
	if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>) {
		if (j == 1) { run_layer_intra_avx512<T, 1>(ptr, begin, end, k); return true; }
		if (j == 2) { run_layer_intra_avx512<T, 2>(ptr, begin, end, k); return true; }
		if (j == 4) { run_layer_intra_avx512<T, 4>(ptr, begin, end, k); return true; }
		if (j == 8) { run_layer_intra_avx512<T, 8>(ptr, begin, end, k); return true; }
		if (j >= 16) { run_layer_inter_avx512<T>(ptr, begin, end, k, j, n); return true; }
	} else if constexpr (std::is_same_v<T, double>) {
		if (j == 1) { run_layer_intra_avx512<T, 1>(ptr, begin, end, k); return true; }
		if (j == 2) { run_layer_intra_avx512<T, 2>(ptr, begin, end, k); return true; }
		if (j == 4) { run_layer_intra_avx512<T, 4>(ptr, begin, end, k); return true; }
		if (j >= 8) { run_layer_inter_avx512<T>(ptr, begin, end, k, j, n); return true; }
	}

	return false;
}
#else
template <typename T>
bool try_run_avx512_layer(std::vector<T>&, std::size_t, std::size_t, std::size_t, std::size_t, bool) {
	return false;
}
#endif

class SpinBarrier {
  public:
	explicit SpinBarrier(std::size_t participants) : threshold(participants), count(participants), generation(0) {
	}

	void wait() {
		const std::size_t gen = generation.load(std::memory_order_acquire);

		if (count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			count.store(threshold, std::memory_order_relaxed);
			generation.store(gen + 1, std::memory_order_release);
		} else {
			while (generation.load(std::memory_order_acquire) == gen) {
#if defined(__x86_64__) || defined(__i386__)
				_mm_pause();
#else
				std::this_thread::yield();
#endif
			}
		}
	}

  private:
	std::size_t threshold;
	std::atomic<std::size_t> count;
	std::atomic<std::size_t> generation;
};

} // namespace

template <typename T>
void run_topk(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
			  const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers) {
	const std::size_t n = data.size();
	SpinBarrier barrier(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers);

	for (std::size_t tid = 0; tid < workers; tid++) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;

			for (std::size_t layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
				const std::size_t k = layers[layer_idx].k;
				const std::size_t j = layers[layer_idx].j;

				// use AVX-512 if possible
				if (try_run_avx512_layer(data, begin, end, k, j, trunc)) {
					barrier.wait();
					continue;
				}

				// fallback scalar loop
				std::size_t i = begin;
				while (i < end) {
					if ((i & j) != 0) {
						i = (i | ((j << 1) - 1)) + 1;
						continue;
					}

					std::size_t chunk_end = std::min((i | (j - 1)) + 1, end);
					if (chunk_end > n) {
						chunk_end = n;
					}

					for (; i < chunk_end; ++i) {
						const std::size_t ixj = i + j;
						
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
				}

				barrier.wait();
			}
		});
	}

	for (auto& t : pool) {
		t.join();
	}
}

template void run_topk<std::int32_t>(std::vector<std::int32_t>& data, const std::vector<common::bitonic::Layer>& layers,
									 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
									 std::size_t workers);
template void run_topk<std::uint32_t>(std::vector<std::uint32_t>& data,
									  const std::vector<common::bitonic::Layer>& layers,
									  const std::vector<std::vector<unsigned char>>& keep, bool trunc,
									  std::size_t workers);
template void run_topk<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
							  const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers);
template void run_topk<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers,
							   const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template void run_topk<_Float16>(std::vector<_Float16>& data, const std::vector<common::bitonic::Layer>& layers,
								 const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers);
#endif

} // namespace cpu::bitonic
