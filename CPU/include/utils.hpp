#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "layers.hpp"

std::size_t count_full_comparators(const std::vector<Layer>& layers, std::size_t n);

std::size_t count_truncated_comparators(const std::vector<Layer>& layers,
										const std::vector<std::vector<unsigned char>>& keep, std::size_t n);

template <typename T>
std::vector<T> generate_random_input(std::size_t n, std::uint64_t seed, int min_value, int max_value);
