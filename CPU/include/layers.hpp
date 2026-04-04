#pragma once

#include <cstddef>
#include <vector>

struct Layer {
	std::size_t k;
	std::size_t j;
};

std::vector<Layer> build_layers(std::size_t n);
