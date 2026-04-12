#pragma once

#include <cstddef>
#include <vector>

namespace common::bitonic {

struct Layer {
	std::size_t k;
	std::size_t j;
};

std::vector<Layer> build_layers(std::size_t n);

std::vector<std::vector<unsigned char>> build_masks(const std::vector<Layer>& layers, std::size_t n, std::size_t topk);

std::size_t count_full_comparators(const std::vector<Layer>& layers, std::size_t n);

std::size_t count_trunc_comparators(const std::vector<Layer>& layers,
									const std::vector<std::vector<unsigned char>>& keep, std::size_t n);

} // namespace common::bitonic
