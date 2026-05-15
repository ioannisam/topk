#pragma once

#include <cstddef>
#include <vector>

#include "common/bitonic.hpp"

namespace cpu::bitonic {

template <typename T>
void run_topk(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers);

} // namespace cpu::bitonic

namespace cpu::map_reduce {

struct RunStats {
	std::size_t tiles_used = 0;
	std::size_t aggregated_candidates = 0;
};

template <typename T>
std::vector<T> run_topk(const std::vector<T>& data, std::size_t k, bool want_max, std::size_t workers,
						RunStats* stats = nullptr);

} // namespace cpu::map_reduce

namespace cpu::ground_truth {

template <typename T>
void run_topk(std::vector<T>& data, std::size_t k, bool want_max);

} // namespace cpu::ground_truth
