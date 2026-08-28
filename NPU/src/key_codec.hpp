#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace npu::utils {

template <typename T> inline std::int32_t to_key(T v) {
	if constexpr (std::is_same_v<T, std::int32_t>) {
		return v;
	} else if constexpr (std::is_same_v<T, std::uint32_t>) {
		return static_cast<std::int32_t>(v ^ 0x80000000u);
	} else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
		std::uint32_t u;
		std::memcpy(&u, &v, 4);
		const std::uint32_t key = (u & 0x80000000u) ? (u ^ 0x7FFFFFFFu) : u;
		return static_cast<std::int32_t>(key);
	} else if constexpr (sizeof(T) == 2) {
		// half: widening to float is exact, so reuse the float key.
		return to_key<float>(static_cast<float>(v));
	} else if constexpr (sizeof(T) == 8) {
		// double: full 64-bit radix key, then keep the high 32 bits.
		std::uint64_t u;
		std::memcpy(&u, &v, 8);
		const std::uint64_t mono =
			(u & 0x8000000000000000ull) ? (u ^ 0xFFFFFFFFFFFFFFFFull) : (u | 0x8000000000000000ull);
		const std::uint32_t hi = static_cast<std::uint32_t>(mono >> 32);
		return static_cast<std::int32_t>(hi ^ 0x80000000u);
	} else {
		return static_cast<std::int32_t>(v);
	}
}

template <typename T> inline void encode_keys(std::int32_t* dst, const T* src, std::size_t n) {
	std::size_t i = 0;
#if defined(__FLT16_MANT_DIG__) && defined(__F16C__)
	if constexpr (std::is_same_v<T, _Float16>) {
		const __m256i flip = _mm256_set1_epi32(0x7FFFFFFF);
		for (; i + 8 <= n; i += 8) {
			const __m256i u =
				_mm256_castps_si256(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
			const __m256i mask = _mm256_srai_epi32(u, 31);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_xor_si256(u, _mm256_and_si256(mask, flip)));
		}
	}
#endif
	for (; i < n; i++) {
		dst[i] = to_key<T>(src[i]);
	}
}

template <typename T> inline T from_key(std::int32_t key) {
	if constexpr (std::is_same_v<T, std::int32_t>) {
		return key;
	} else if constexpr (std::is_same_v<T, std::uint32_t>) {
		return static_cast<std::uint32_t>(key) ^ 0x80000000u;
	} else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
		const std::uint32_t key_u = static_cast<std::uint32_t>(key);
		const std::uint32_t u = (key_u & 0x80000000u) ? (key_u ^ 0x7FFFFFFFu) : key_u;
		T v;
		std::memcpy(&v, &u, 4);
		return v;
	} else if constexpr (sizeof(T) == 2) {
		return static_cast<T>(from_key<float>(key));
	} else {
		static_assert(
			sizeof(T) != 8,
			"from_key<T> cannot recover a double from its lossy 32-bit key; "
			"the double path must re-derive keys from the original data instead"
		);
		return static_cast<T>(key);
	}
}

} // namespace npu::utils
