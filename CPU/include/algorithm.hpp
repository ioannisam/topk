#pragma once

#include <cstddef>
#include <vector>

#include "layers.hpp"

std::vector<std::vector<unsigned char>> build_masks(const std::vector<Layer>& layers, std::size_t n, std::size_t k);

void run_network_parallel(std::vector<int>& data, const std::vector<Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool truncated, std::size_t workers);
