#pragma once

#include <cstddef>
#include <vector>

#include "layers.hpp"

std::vector<std::vector<unsigned char>> build_masks(const std::vector<Layer>& layers, std::size_t n, std::size_t k);

template <typename T>
void run_network_parallel(std::vector<T>& data, const std::vector<Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers);
