#include "utils.hpp"

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

std::size_t count_full_comparators(const std::vector<Layer>& layers, std::size_t n) {
	return layers.size() * (n / 2);
}

std::size_t count_trunc_comparators(const std::vector<Layer>& layers,
									const std::vector<std::vector<unsigned char>>& keep, std::size_t n) {

	std::size_t active = 0;
	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const std::size_t j = layers[layer_idx].j;
		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t ixj = i ^ j;
			if (ixj <= i || ixj >= n) {
				continue;
			}
			if (keep[layer_idx][i] || keep[layer_idx][ixj]) {
				active++;
			}
		}
	}
	return active;
}

template <typename T>
std::vector<T> generate_random_input(std::size_t n, std::uint64_t seed, int min_value, int max_value) {
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	std::vector<T> input(n);
	if constexpr (std::is_integral_v<T>) {
		if constexpr (std::is_unsigned_v<T>) {
			const auto umin = static_cast<unsigned long long>(std::max(0, min_value));
			const auto umax = static_cast<unsigned long long>(std::max(0, max_value));
			std::uniform_int_distribution<unsigned long long> dist(umin, umax);
			for (std::size_t i = 0; i < n; i++) {
				input[i] = static_cast<T>(dist(rng));
			}
		} else {
			std::uniform_int_distribution<long long> dist(min_value, max_value);
			for (std::size_t i = 0; i < n; i++) {
				input[i] = static_cast<T>(dist(rng));
			}
		}
	} else {
		std::uniform_real_distribution<double> dist(static_cast<double>(min_value), static_cast<double>(max_value));
		for (std::size_t i = 0; i < n; i++) {
			input[i] = static_cast<T>(dist(rng));
		}
	}

	return input;
}

template std::vector<std::int32_t> generate_random_input<std::int32_t>(std::size_t n, std::uint64_t seed, int min_value,
																	   int max_value);
template std::vector<std::uint32_t> generate_random_input<std::uint32_t>(std::size_t n, std::uint64_t seed,
																		 int min_value, int max_value);
template std::vector<float> generate_random_input<float>(std::size_t n, std::uint64_t seed, int min_value,
														 int max_value);
template std::vector<double> generate_random_input<double>(std::size_t n, std::uint64_t seed, int min_value,
														   int max_value);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> generate_random_input<_Float16>(std::size_t n, std::uint64_t seed, int min_value,
															   int max_value);
#endif
