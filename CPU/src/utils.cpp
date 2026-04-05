#include "utils.hpp"

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
