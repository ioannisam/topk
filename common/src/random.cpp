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

constexpr std::size_t kAdversarialSpikes = 4096;
constexpr double kTrimodalCenters[] = {0.2, 0.5, 0.8};

double sample_clustered(std::mt19937& rng, double lo, double hi, common::config::Distribution dist) {
	const double span = hi - lo;
	double mean = 0.5 * (lo + hi);
	double sigma = span > 0.0 ? span / 6.0 : 1.0;

	if (dist == common::config::Distribution::Trimodal) {
		std::uniform_int_distribution<int> mode(0, 2);
		mean = lo + span * kTrimodalCenters[mode(rng)];
		sigma = span > 0.0 ? span / 30.0 : 1.0;
	}

	std::normal_distribution<double> gauss(mean, sigma);
	return std::clamp(gauss(rng), lo, hi);
}

bool is_clustered(common::config::Distribution dist) {
	return dist == common::config::Distribution::Normal || dist == common::config::Distribution::Trimodal;
}

template <typename T>
T sample_value(std::mt19937& rng, int min_value, int max_value, common::config::Distribution dist) {
	if constexpr (std::is_integral_v<T>) {
		long long lo = min_value;
		long long hi = max_value;
		if constexpr (std::is_unsigned_v<T>) {
			lo = std::max<long long>(0, lo);
			hi = std::max<long long>(0, hi);
		}

		if (is_clustered(dist)) {
			const double value = sample_clustered(rng, static_cast<double>(lo), static_cast<double>(hi), dist);
			return static_cast<T>(std::llround(value));
		}

		std::uniform_int_distribution<long long> uniform(lo, hi);
		return static_cast<T>(uniform(rng));
	}

	const double eff_min = static_cast<double>(min_value);
	const double eff_max = static_cast<double>(max_value);

	if (is_clustered(dist)) {
		return static_cast<T>(sample_clustered(rng, eff_min, eff_max, dist));
	}

	std::uniform_real_distribution<double> uniform(eff_min, eff_max);
	return static_cast<T>(uniform(rng));
}

template <typename T> void apply_ascending_spikes(std::vector<T>& input, int low, int high) {
	const std::size_t n = input.size();
	double lo = static_cast<double>(low);
	double hi = static_cast<double>(high);
	if constexpr (std::is_unsigned_v<T>) {
		lo = std::max(0.0, lo);
		hi = std::max(0.0, hi);
	}
	if (n == 0 || hi <= lo) {
		return;
	}

	const std::size_t spikes = std::min<std::size_t>(kAdversarialSpikes, n);
	const std::size_t stride = n / spikes;
	const double span = hi - lo;

	for (std::size_t s = 0; s < spikes; s++) {
		const double frac = static_cast<double>(s + 1) / static_cast<double>(spikes);
		const double value = lo + span * frac;
		if constexpr (std::is_integral_v<T>) {
			input[s * stride] = static_cast<T>(std::llround(std::min(value, hi)));
		} else {
			input[s * stride] = static_cast<T>(std::min(value, hi));
		}
	}
}

} // namespace

template <typename T>
std::vector<T> generate_random_input(
	std::size_t n, std::uint64_t seed, int min_value, int max_value, common::config::Distribution dist
) {
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));

	const bool adversarial = dist == common::config::Distribution::Adversarial;
	const int base_max = adversarial
							 ? static_cast<int>(0.5 * (static_cast<double>(min_value) + static_cast<double>(max_value)))
							 : max_value;

	std::vector<T> input(n);
	for (std::size_t i = 0; i < n; i++) {
		input[i] = sample_value<T>(rng, min_value, base_max, dist);
	}

	if (dist == common::config::Distribution::Sorted) {
		std::sort(input.begin(), input.end(), std::less<T>());
	} else if (dist == common::config::Distribution::Reverse) {
		std::sort(input.begin(), input.end(), std::greater<T>());
	} else if (adversarial) {
		apply_ascending_spikes<T>(input, base_max, max_value);
	}

	return input;
}

template std::vector<std::int32_t> generate_random_input<std::int32_t>(
	std::size_t n, std::uint64_t seed, int min_value, int max_value, common::config::Distribution dist
);
template std::vector<std::uint32_t> generate_random_input<std::uint32_t>(
	std::size_t n, std::uint64_t seed, int min_value, int max_value, common::config::Distribution dist
);
template std::vector<float> generate_random_input<float>(
	std::size_t n, std::uint64_t seed, int min_value, int max_value, common::config::Distribution dist
);
template std::vector<double> generate_random_input<double>(
	std::size_t n, std::uint64_t seed, int min_value, int max_value, common::config::Distribution dist
);

#if defined(__FLT16_MANT_DIG__)
template std::vector<_Float16> generate_random_input<_Float16>(
	std::size_t n, std::uint64_t seed, int min_value, int max_value, common::config::Distribution dist
);
#endif

} // namespace common::utils
