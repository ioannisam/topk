#include "common/bitonic.hpp"

namespace common::bitonic {

std::vector<Layer> build_layers(std::size_t n, std::size_t topk) {
	std::vector<Layer> layers;
	if (topk > n) {
		topk = n;
	}
	if (topk == 0) {
		return layers;
	}

	std::size_t topk_pow2 = 1;
	while (topk_pow2 < topk) {
		topk_pow2 <<= 1;
	}
	if (topk_pow2 > n) {
		topk_pow2 = n;
	}

	std::size_t current_n = n;

	// standard bitonic sort up to blocks of size topk_pow2
	for (std::size_t stage = 2; stage <= topk_pow2; stage <<= 1) {
		for (std::size_t stride = stage >> 1; stride > 0; stride >>= 1) {
			layers.push_back({LayerType::Normal, current_n, stage, stride});
		}
	}

	// truncated merges
	while (current_n > topk_pow2) {
		layers.push_back({LayerType::Truncate, current_n, topk_pow2 * 2, topk_pow2});

		current_n /= 2;
		for (std::size_t stride = topk_pow2 >> 1; stride > 0; stride >>= 1) {
			layers.push_back({LayerType::Normal, current_n, topk_pow2, stride});
		}
	}

	return layers;
}

std::size_t count_full_comparators(std::size_t n) {
	std::size_t num_layers = 0;
	for (std::size_t stage = 2; stage <= n; stage <<= 1) {
		for (std::size_t stride = stage >> 1; stride > 0; stride >>= 1) {
			num_layers++;
		}
	}
	return num_layers * (n / 2);
}

std::size_t count_trunc_comparators(const std::vector<Layer>& layers) {
	std::size_t active = 0;
	for (const auto& layer : layers) {
		active += layer.active_n / 2;
	}
	return active;
}

} // namespace common::bitonic
