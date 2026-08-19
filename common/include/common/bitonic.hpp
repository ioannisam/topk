#pragma once

#include <cstddef>
#include <vector>

namespace common::bitonic {

enum class LayerType { Normal, Truncate };

struct Layer {
	LayerType type;
	std::size_t active_n;
	std::size_t stage;
	std::size_t stride;
};

std::vector<Layer> build_layers(std::size_t n, std::size_t topk);

std::size_t count_full_comparators(std::size_t n);

std::size_t count_trunc_comparators(const std::vector<Layer>& layers);

} // namespace common::bitonic
