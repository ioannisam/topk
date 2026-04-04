#include "utils.hpp"

#include <random>

std::size_t count_full_comparators(const std::vector<Layer>& layers, std::size_t n) {
	return layers.size() * (n / 2);
}

std::size_t count_truncated_comparators(const std::vector<Layer>& layers,
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

std::vector<int> generate_random_input(std::size_t n, std::uint64_t seed, int min_value, int max_value) {
	std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
	std::uniform_int_distribution<int> dist(min_value, max_value);

	std::vector<int> input(n);
	for (std::size_t i = 0; i < n; i++) {
		input[i] = dist(rng);
	}

	return input;
}
