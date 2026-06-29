#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/config.hpp"

namespace common::utils {

template <typename T>
std::vector<T> generate_random_input(std::size_t n, std::uint64_t seed, int min_value, int max_value,
									 common::config::Distribution dist = common::config::Distribution::Uniform);

} // namespace common::utils
