#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <immintrin.h>
#include <type_traits>
#include <utility>

namespace cpu::simd {

#if defined(__x86_64__) || defined(__i386__)
inline bool cpu_supports_avx512f() {
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

inline bool cpu_supports_avx2() {
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

inline bool use_avx512() {
	static const bool enabled = []() {
		if (!cpu_supports_avx512f()) {
			return false;
		}
		const char* env = std::getenv("TOPK_FORCE_AVX512");
		return env == nullptr || env[0] != '0';
	}();
	return enabled;
}

template <typename T> struct SimdTraits512;
template <typename T> struct SimdTraits256;

// ==========================================
// AVX-512 Traits
// ==========================================

#if defined(__AVX512F__)

template <> struct SimdTraits512<float> {
	using Vec = __m512;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	// -- Bitonic Sort Primitives --
	static Vec load(const float* p) {
		return _mm512_loadu_ps(p);
	}
	static void store(float* p, Vec v) {
		_mm512_storeu_ps(p, v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm512_min_ps(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm512_max_ps(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm512_mask_blend_ps(m, a, b);
	}

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) {
			return _mm512_permutexvar_ps(_mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14), v);
		}
		if constexpr (J == 2) {
			return _mm512_permutexvar_ps(_mm512_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13), v);
		}
		if constexpr (J == 4) {
			return _mm512_permutexvar_ps(_mm512_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11), v);
		}
		if constexpr (J == 8) {
			return _mm512_permutexvar_ps(_mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7), v);
		}
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		const __m512i lane_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		const __m512i base = _mm512_add_epi32(_mm512_set1_epi32(static_cast<std::int32_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(_mm512_set1_epi32(J), base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, _mm512_set1_epi32(static_cast<std::int32_t>(k)));
		const Mask desc_mask = _mm512_cmpneq_epi32_mask(dir_bits, _mm512_setzero_si512());
		Mask odd_mask = (J == 1) ? 0xAAAA : (J == 2) ? 0xCCCC : (J == 4) ? 0xF0F0 : 0xFF00;
		return odd_mask ^ desc_mask;
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const float* ptr, float threshold) {
		const __m512 v = _mm512_loadu_ps(ptr);
		const __m512 t = _mm512_set1_ps(threshold);
		return WantMax ? _mm512_cmp_ps_mask(v, t, _CMP_GT_OQ) : _mm512_cmp_ps_mask(v, t, _CMP_LT_OQ);
	}
};

template <> struct SimdTraits512<std::int32_t> {
	using Vec = __m512i;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	// -- Bitonic Sort Primitives --
	static Vec load(const std::int32_t* p) {
		return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p));
	}
	static void store(std::int32_t* p, Vec v) {
		_mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm512_min_epi32(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm512_max_epi32(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm512_mask_blend_epi32(m, a, b);
	}

	template <int J> static Vec permutex(Vec v) {
		__m512 tmp = SimdTraits512<float>::template permutex<J>(_mm512_castsi512_ps(v));
		return _mm512_castps_si512(tmp);
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		return SimdTraits512<float>::template get_blend_mask<J>(i, k);
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const std::int32_t* ptr, std::int32_t threshold) {
		const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
		const __m512i t = _mm512_set1_epi32(threshold);
		return WantMax ? _mm512_cmpgt_epi32_mask(v, t) : _mm512_cmpgt_epi32_mask(t, v);
	}
};

template <> struct SimdTraits512<std::uint32_t> {
	using Vec = __m512i;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	// -- Bitonic Sort Primitives --
	static Vec load(const std::uint32_t* p) {
		return _mm512_loadu_si512(reinterpret_cast<const __m512i*>(p));
	}
	static void store(std::uint32_t* p, Vec v) {
		_mm512_storeu_si512(reinterpret_cast<__m512i*>(p), v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm512_min_epu32(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm512_max_epu32(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm512_mask_blend_epi32(m, a, b);
	}
	template <int J> static Vec permutex(Vec v) {
		return SimdTraits512<std::int32_t>::template permutex<J>(v);
	}
	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		return SimdTraits512<float>::template get_blend_mask<J>(i, k);
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const std::uint32_t* ptr, std::uint32_t threshold) {
		const __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(ptr));
		const __m512i t = _mm512_set1_epi32(static_cast<std::int32_t>(threshold));
		return WantMax ? _mm512_cmp_epu32_mask(v, t, _MM_CMPINT_GT) : _mm512_cmp_epu32_mask(t, v, _MM_CMPINT_GT);
	}
};

template <> struct SimdTraits512<double> {
	using Vec = __m512d;
	using Mask = __mmask8;
	static constexpr std::size_t width = 8;

	// -- Bitonic Sort Primitives --
	static Vec load(const double* p) {
		return _mm512_loadu_pd(p);
	}
	static void store(double* p, Vec v) {
		_mm512_storeu_pd(p, v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm512_min_pd(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm512_max_pd(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm512_mask_blend_pd(m, a, b);
	}

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) {
			return _mm512_permutexvar_pd(_mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6), v);
		}
		if constexpr (J == 2) {
			return _mm512_permutexvar_pd(_mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5), v);
		}
		if constexpr (J == 4) {
			return _mm512_permutexvar_pd(_mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3), v);
		}
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		const __m512i lane_idx = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
		const __m512i base = _mm512_add_epi64(_mm512_set1_epi64(static_cast<std::int64_t>(i)), lane_idx);
		const __m512i pair_base = _mm512_andnot_si512(_mm512_set1_epi64(J), base);
		const __m512i dir_bits = _mm512_and_si512(pair_base, _mm512_set1_epi64(static_cast<std::int64_t>(k)));
		const Mask desc_mask = _mm512_cmpneq_epi64_mask(dir_bits, _mm512_setzero_si512());
		Mask odd_mask = (J == 1) ? 0xAA : (J == 2) ? 0xCC : 0xF0;
		return odd_mask ^ desc_mask;
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const double* ptr, double threshold) {
		const __m512d v = _mm512_loadu_pd(ptr);
		const __m512d t = _mm512_set1_pd(threshold);
		return WantMax ? _mm512_cmp_pd_mask(v, t, _CMP_GT_OQ) : _mm512_cmp_pd_mask(v, t, _CMP_LT_OQ);
	}
};

#if defined(__FLT16_MANT_DIG__)
template <> struct SimdTraits512<_Float16> {
	using Vec = __m512;
	using Mask = __mmask16;
	static constexpr std::size_t width = 16;

	// -- Bitonic Sort Primitives --
	static Vec load(const _Float16* p) {
		return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
	}
	static void store(_Float16* p, Vec v) {
		_mm256_storeu_si256(
			reinterpret_cast<__m256i*>(p), _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
		);
	}
	static Vec min(Vec a, Vec b) {
		return _mm512_min_ps(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm512_max_ps(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm512_mask_blend_ps(m, a, b);
	}

	template <int J> static Vec permutex(Vec v) {
		return SimdTraits512<float>::template permutex<J>(v);
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		return SimdTraits512<float>::template get_blend_mask<J>(i, k);
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const _Float16* ptr, _Float16 threshold) {
		const __m512 v = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
		const __m512 t = _mm512_set1_ps(static_cast<float>(threshold));
		return WantMax ? _mm512_cmp_ps_mask(v, t, _CMP_GT_OQ) : _mm512_cmp_ps_mask(v, t, _CMP_LT_OQ);
	}
};
#endif

#endif // __AVX512F__

// ==========================================
// AVX2 Traits
// ==========================================

#if defined(__AVX2__)

template <> struct SimdTraits256<float> {
	using Vec = __m256;
	using Mask = __m256;
	static constexpr std::size_t width = 8;

	// -- Bitonic Sort Primitives --
	static Vec load(const float* p) {
		return _mm256_loadu_ps(p);
	}
	static void store(float* p, Vec v) {
		_mm256_storeu_ps(p, v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm256_min_ps(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm256_max_ps(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm256_blendv_ps(a, b, m);
	}

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) {
			return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
		}
		if constexpr (J == 2) {
			return _mm256_shuffle_ps(v, v, _MM_SHUFFLE(1, 0, 3, 2));
		}
		if constexpr (J == 4) {
			return _mm256_permute2f128_ps(v, v, 0x01);
		}
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		alignas(32) std::int32_t mask_arr[8];
		for (int l = 0; l < 8; l++) {
			std::size_t curr = i + l;
			bool desc = ((curr & ~static_cast<std::size_t>(J)) & k) != 0;
			bool odd = (curr & J) != 0;
			mask_arr[l] = (odd ^ desc) ? 0xFFFFFFFF : 0;
		}
		return _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask_arr)));
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const float* ptr, float threshold) {
		const __m256 v = _mm256_loadu_ps(ptr);
		const __m256 t = _mm256_set1_ps(threshold);
		const __m256 cmp = WantMax ? _mm256_cmp_ps(v, t, _CMP_GT_OQ) : _mm256_cmp_ps(v, t, _CMP_LT_OQ);
		return static_cast<std::uint64_t>(_mm256_movemask_ps(cmp));
	}
};

template <> struct SimdTraits256<std::int32_t> {
	using Vec = __m256i;
	using Mask = __m256i;
	static constexpr std::size_t width = 8;

	// -- Bitonic Sort Primitives --
	static Vec load(const std::int32_t* p) {
		return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
	}
	static void store(std::int32_t* p, Vec v) {
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm256_min_epi32(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm256_max_epi32(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm256_blendv_epi8(a, b, m);
	}

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) {
			return _mm256_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1));
		}
		if constexpr (J == 2) {
			return _mm256_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));
		}
		if constexpr (J == 4) {
			return _mm256_permute2x128_si256(v, v, 0x01);
		}
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		alignas(32) std::int32_t mask_arr[8];
		for (int l = 0; l < 8; l++) {
			std::size_t curr = i + l;
			bool desc = ((curr & ~static_cast<std::size_t>(J)) & k) != 0;
			bool odd = (curr & J) != 0;
			mask_arr[l] = (odd ^ desc) ? -1 : 0;
		}
		return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask_arr));
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const std::int32_t* ptr, std::int32_t threshold) {
		const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
		const __m256i t = _mm256_set1_epi32(threshold);
		const __m256i cmp = WantMax ? _mm256_cmpgt_epi32(v, t) : _mm256_cmpgt_epi32(t, v);
		return static_cast<std::uint64_t>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
	}
};

template <> struct SimdTraits256<std::uint32_t> {
	using Vec = __m256i;
	using Mask = __m256i;
	static constexpr std::size_t width = 8;

	// -- Bitonic Sort Primitives --
	static Vec load(const std::uint32_t* p) {
		return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
	}
	static void store(std::uint32_t* p, Vec v) {
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm256_min_epu32(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm256_max_epu32(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm256_blendv_epi8(a, b, m);
	}
	template <int J> static Vec permutex(Vec v) {
		return SimdTraits256<std::int32_t>::template permutex<J>(v);
	}
	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		return SimdTraits256<std::int32_t>::template get_blend_mask<J>(i, k);
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const std::uint32_t* ptr, std::uint32_t threshold) {
		const __m256i sign = _mm256_set1_epi32(static_cast<std::int32_t>(0x80000000u));
		const __m256i v = _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)), sign);
		const __m256i t = _mm256_xor_si256(_mm256_set1_epi32(static_cast<std::int32_t>(threshold)), sign);
		const __m256i cmp = WantMax ? _mm256_cmpgt_epi32(v, t) : _mm256_cmpgt_epi32(t, v);
		return static_cast<std::uint64_t>(_mm256_movemask_ps(_mm256_castsi256_ps(cmp)));
	}
};

template <> struct SimdTraits256<double> {
	using Vec = __m256d;
	using Mask = __m256d;
	static constexpr std::size_t width = 4;

	// -- Bitonic Sort Primitives --
	static Vec load(const double* p) {
		return _mm256_loadu_pd(p);
	}
	static void store(double* p, Vec v) {
		_mm256_storeu_pd(p, v);
	}
	static Vec min(Vec a, Vec b) {
		return _mm256_min_pd(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm256_max_pd(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm256_blendv_pd(a, b, m);
	}

	template <int J> static Vec permutex(Vec v) {
		if constexpr (J == 1) {
			return _mm256_permute_pd(v, 0x5);
		}
		if constexpr (J == 2) {
			return _mm256_permute2f128_pd(v, v, 0x01);
		}
		return v;
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		alignas(32) std::int64_t mask_arr[4];
		for (int l = 0; l < 4; l++) {
			std::size_t curr = i + l;
			bool desc = ((curr & ~static_cast<std::size_t>(J)) & k) != 0;
			bool odd = (curr & J) != 0;
			mask_arr[l] = (odd ^ desc) ? -1 : 0;
		}
		return _mm256_castsi256_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask_arr)));
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const double* ptr, double threshold) {
		const __m256d v = _mm256_loadu_pd(ptr);
		const __m256d t = _mm256_set1_pd(threshold);
		const __m256d cmp = WantMax ? _mm256_cmp_pd(v, t, _CMP_GT_OQ) : _mm256_cmp_pd(v, t, _CMP_LT_OQ);
		return static_cast<std::uint64_t>(_mm256_movemask_pd(cmp));
	}
};

#if defined(__FLT16_MANT_DIG__) && defined(__F16C__)
template <> struct SimdTraits256<_Float16> {
	using Vec = __m256;
	using Mask = __m256;
	static constexpr std::size_t width = 8;

	// -- Bitonic Sort Primitives --
	static Vec load(const _Float16* p) {
		return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
	}
	static void store(_Float16* p, Vec v) {
		_mm_storeu_si128(
			reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
		);
	}
	static Vec min(Vec a, Vec b) {
		return _mm256_min_ps(a, b);
	}
	static Vec max(Vec a, Vec b) {
		return _mm256_max_ps(a, b);
	}
	static Vec blend(Mask m, Vec a, Vec b) {
		return _mm256_blendv_ps(a, b, m);
	}

	template <int J> static Vec permutex(Vec v) {
		return SimdTraits256<float>::template permutex<J>(v);
	}

	template <int J> static Mask get_blend_mask(std::size_t i, std::size_t k) {
		return SimdTraits256<float>::template get_blend_mask<J>(i, k);
	}

	// -- Map-Reduce Primitives --
	template <bool WantMax> static std::uint64_t get_candidate_mask(const _Float16* ptr, _Float16 threshold) {
		const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
		const __m256 t = _mm256_set1_ps(static_cast<float>(threshold));
		const __m256 cmp = WantMax ? _mm256_cmp_ps(v, t, _CMP_GT_OQ) : _mm256_cmp_ps(v, t, _CMP_LT_OQ);
		return static_cast<std::uint64_t>(_mm256_movemask_ps(cmp));
	}
};
#endif

#endif // __AVX2__

// Detection helpers to avoid instantiating trait specializations the target cannot provide
template <typename T, typename = void> struct has_simd512_width : std::false_type {};
template <typename T> struct has_simd512_width<T, std::void_t<decltype(SimdTraits512<T>::width)>> : std::true_type {};

template <typename T, typename = void> struct has_simd256_width : std::false_type {};
template <typename T> struct has_simd256_width<T, std::void_t<decltype(SimdTraits256<T>::width)>> : std::true_type {};

template <bool WantMax, typename T>
inline std::uint64_t get_candidate_mask_simd(const T* ptr, T threshold, bool use_512, bool use_256) {
	if (use_512) {
		if constexpr (has_simd512_width<T>::value) {
			return SimdTraits512<T>::template get_candidate_mask<WantMax>(ptr, threshold);
		}
	}
	if (use_256) {
		if constexpr (has_simd256_width<T>::value) {
			return SimdTraits256<T>::template get_candidate_mask<WantMax>(ptr, threshold);
		}
	}
	return 0;
}

template <typename T> inline std::size_t simd_block_width(bool use_512, bool use_256) {
	if (use_512) {
		if constexpr (has_simd512_width<T>::value) {
			return SimdTraits512<T>::width;
		}
	}
	if (use_256) {
		if constexpr (has_simd256_width<T>::value) {
			return SimdTraits256<T>::width;
		}
	}
	return 1;
}

#else
// for non-x86_64 targets
inline bool use_avx512() {
	return false;
}
inline bool cpu_supports_avx2() {
	return false;
}
template <bool WantMax, typename T> inline std::uint64_t get_candidate_mask_simd(const T*, T, bool, bool) {
	return 0;
}
template <typename T> inline std::size_t simd_block_width(bool, bool) {
	return 1;
}
#endif

} // namespace cpu::simd
