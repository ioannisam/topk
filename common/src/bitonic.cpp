#include "common/bitonic.hpp"

namespace common::bitonic {

std::vector<Layer> build_layers(std::size_t n) {
	std::vector<Layer> layers;
	for (std::size_t k = 2; k <= n; k <<= 1) {
		for (std::size_t j = k >> 1; j > 0; j >>= 1) {
			layers.push_back(Layer{k, j});
		}
	}
	return layers;
}

std::vector<std::vector<unsigned char>> build_masks(const std::vector<Layer>& layers, std::size_t n, std::size_t topk) {
	std::vector<std::vector<unsigned char>> keep(layers.size(), std::vector<unsigned char>(n, 0));
	std::vector<unsigned char> needed(n, 0);
	for (std::size_t i = 0; i < topk; i++) {
		needed[i] = 1;
	}

	for (std::size_t idx = layers.size(); idx-- > 0;) {
		const std::size_t j = layers[idx].j;
		std::vector<unsigned char> prev_needed = needed;

		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t ixj = i ^ j;
			if (ixj <= i) {
				continue;
			}
			if (needed[i] || needed[ixj]) {
				keep[idx][i] = 1;
				keep[idx][ixj] = 1;
				prev_needed[i] = 1;
				prev_needed[ixj] = 1;
			}
		}

		needed.swap(prev_needed);
	}

	return keep;
}

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

} // namespace common::bitonic
