#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace common {

template <typename T>
std::vector<T> generate_random_input(std::size_t n, std::uint64_t seed, int min_value, int max_value);

} // namespace common
