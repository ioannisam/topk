#pragma once

#include <cstddef>
#include <vector>

#include "common/bitonic.hpp"

namespace cpu::bitonic {

template <typename T>
void run_network_parallel(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers);

} // namespace cpu::bitonic
