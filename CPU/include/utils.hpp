#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "layers.hpp"

std::size_t count_full_comparators(const std::vector<Layer>& layers, std::size_t n);

std::size_t count_trunc_comparators(const std::vector<Layer>& layers,
									const std::vector<std::vector<unsigned char>>& keep, std::size_t n);
