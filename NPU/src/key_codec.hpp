#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

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
		// fp16 (_Float16): widening to float is exact, so reuse the float key.
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
	} else if constexpr (sizeof(T) == 8) {
		const std::uint64_t mono = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key) ^ 0x80000000u) << 32;
		const std::uint64_t u = (mono & 0x8000000000000000ull) ? (mono & 0x7FFFFFFFFFFFFFFFull) : (~mono);
		T v;
		std::memcpy(&v, &u, 8);
		return v;
	} else {
		return static_cast<T>(key);
	}
}

} // namespace npu::utils
