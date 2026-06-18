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
	} else {
		return static_cast<T>(key);
	}
}

} // namespace npu::utils
