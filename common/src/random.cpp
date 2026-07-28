#include "common/random.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>

namespace common::utils {

namespace {

template <typename T> T sample_value(std::mt19937& rng, int min_value, int max_value, bool normal) {
	if constexpr (std::is_integral_v<T>) {
		long long lo = min_value;
		long long hi = max_value;
		if constexpr (std::is_unsigned_v<T>) {
			lo = std::max<long long>(0, lo);
			hi = std::max<long long>(0, hi);
		}

		if (normal) {
			const double mean = 0.5 * (static_cast<double>(lo) + static_cast<double>(hi));
			const double span = static_cast<double>(hi) - static_cast<double>(lo);
			const double sigma = span > 0.0 ? span / 6.0 : 1.0;
			std::normal_distribution<double> dist(mean, sigma);
			double value = dist(rng);
			value = std::min(std::max(value, static_cast<double>(lo)), static_cast<double>(hi));
			return static_cast<T>(std::llround(value));
		}

		std::uniform_int_distribution<long long> dist(lo, hi);
		return static_cast<T>(dist(rng));
	}

	const double eff_min = static_cast<double>(min_value);
	const double eff_max = static_cast<double>(max_value);

	if (normal) {
		const double mean = 0.5 * (eff_min + eff_max);
		const double span = eff_max - eff_min;
		const double sigma = span > 0.0 ? span / 6.0 : 1.0;
		std::normal_distribution<double> dist(mean, sigma);
		double value = dist(rng);
		value = std::min(std::max(value, eff_min), eff_max);
		return static_cast<T>(value);
	}

	std::uniform_real_distribution<double> dist(eff_min, eff_max);
	return static_cast<T>(dist(rng));
}

} // namespace

template <typename T>
std::vector<T> generate_random_input(std::size_t n, std::uint64_t seed, int min_value, int max_value,
									 common::config::Distribution dist) {
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	const bool normal = dist == common::config::Distribution::Normal;
	std::vector<T> input(n);
	for (std::size_t i = 0; i < n; i++) {
		input[i] = sample_value<T>(rng, min_value, max_value, normal);
	}

	if (dist == common::config::Distribution::Sorted) {
		std::sort(input.begin(), input.end(), std::less<T>());
	} else if (dist == common::config::Distribution::Reverse) {
		std::sort(input.begin(), input.end(), std::greater<T>());
	}

	return input;
}

template std::vector<std::int32_t> generate_random_input<std::int32_t>(std::size_t n, std::uint64_t seed, int min_value,
																	   int max_value,
																	   common::config::Distribution dist);
template std::vector<std::uint32_t> generate_random_input<std::uint32_t>(std::size_t n, std::uint64_t seed,
																		 int min_value, int max_value,
																		 common::config::Distribution dist);
template std::vector<float> generate_random_input<float>(std::size_t n, std::uint64_t seed, int min_value,
														 int max_value, common::config::Distribution dist);
template std::vector<double> generate_random_input<double>(std::size_t n, std::uint64_t seed, int min_value,
														   int max_value, common::config::Distribution dist);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> generate_random_input<_Float16>(std::size_t n, std::uint64_t seed, int min_value,
															   int max_value, common::config::Distribution dist);
#endif

} // namespace common::utils
