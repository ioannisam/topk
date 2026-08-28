#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "common/bitonic.hpp"

namespace npu {

struct PhaseTimers {
	double sample_ms = 0.0;
	double setup_ms = 0.0;
	double stage_ms = 0.0;
	double dispatch_ms = 0.0;
	double wait_ms = 0.0;
	double merge_ms = 0.0;
	double finalize_ms = 0.0;
};

class PhaseTimer {
  public:
	explicit PhaseTimer(double& sink) : target(sink), mark(std::chrono::high_resolution_clock::now()) {
	}

	~PhaseTimer() {
		target += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - mark).count();
	}

	PhaseTimer(const PhaseTimer&) = delete;
	PhaseTimer& operator=(const PhaseTimer&) = delete;

  private:
	double& target;
	std::chrono::high_resolution_clock::time_point mark;
};

} // namespace npu

namespace npu::bitonic {

struct RunStats {
	double elapsed_ms;
	std::size_t layer_dispatches;
	std::size_t active_comparators;
	std::size_t workers;
	bool used_offload;
	npu::PhaseTimers phases{};
	double bytes_moved = 0.0;
};

std::string query_device_name();
std::string query_device_bdf();
bool is_offload_configured();

template <typename T>
RunStats run_topk(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers);

} // namespace npu::bitonic

namespace npu::map_reduce {

struct RunStats {
	double elapsed_ms;
	std::size_t layer_dispatches;
	bool used_offload;
	npu::PhaseTimers phases{};
	double bytes_moved = 0.0;
	std::size_t tiles_used = 0;
	std::size_t aggregated_candidates = 0;
};

template <typename T>
std::vector<T> run_topk(
	const std::vector<T>& data, std::size_t k, bool want_max, npu::map_reduce::RunStats* stats = nullptr
);

} // namespace npu::map_reduce

namespace npu::ground_truth {

template <typename T>
void run_topk(std::vector<T>& data, std::size_t k, bool want_max);

} // namespace npu::ground_truth
