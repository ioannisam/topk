#include "layers.hpp"

std::vector<Layer> build_layers(std::size_t n) {

	std::vector<Layer> layers;
	for (std::size_t k = 2; k <= n; k <<= 1) {
		for (std::size_t j = k >> 1; j > 0; j >>= 1) {
			layers.push_back(Layer{k, j});
		}
	}

	return layers;
}
