#pragma once

#include <cstddef>
#include <vector>

namespace common::bitonic {

enum class LayerType {
	Normal,	 // Standard bitonic compare-and-swap
	Truncate // Compare elements, discard the losers, and compact the array!
};

struct Layer {
	LayerType type;
	std::size_t active_n; // The number of elements involved in this layer
	std::size_t k;		  // The 'k' parameter (dictates ascending/descending direction)
	std::size_t j;		  // The stride distance
};

std::vector<Layer> build_layers(std::size_t n, std::size_t topk);

std::size_t count_full_comparators(std::size_t n);

std::size_t count_trunc_comparators(const std::vector<Layer>& layers);

} // namespace common::bitonic
