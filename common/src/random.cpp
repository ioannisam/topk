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
		if (normal) {
			const double mean = 0.5 * (static_cast<double>(min_value) + static_cast<double>(max_value));
			const double span = static_cast<double>(max_value) - static_cast<double>(min_value);
			const double sigma = span > 0.0 ? span / 6.0 : 1.0;
			std::normal_distribution<double> dist(mean, sigma);
			double value = dist(rng);
			value = std::min(std::max(value, static_cast<double>(min_value)), static_cast<double>(max_value));
			return static_cast<T>(std::llround(value));
		}

		if constexpr (std::is_unsigned_v<T>) {
			const auto umin = static_cast<unsigned long long>(std::max(0, min_value));
			const auto umax = static_cast<unsigned long long>(std::max(0, max_value));
			std::uniform_int_distribution<unsigned long long> dist(umin, umax);
			return static_cast<T>(dist(rng));
		}
		std::uniform_int_distribution<long long> dist(min_value, max_value);
		return static_cast<T>(dist(rng));
	}

	double eff_min = static_cast<double>(min_value);
	double eff_max = static_cast<double>(max_value);
#if defined(__FLT16_MANT_DIG__)
	if constexpr (std::is_same_v<T, _Float16>) {
		constexpr double kHalfMax = 65504.0;
		eff_min = std::clamp(eff_min, -kHalfMax, kHalfMax);
		eff_max = std::clamp(eff_max, -kHalfMax, kHalfMax);
	}
#endif

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
