#pragma once

#include <cstddef>

#include "common/energy.hpp"

namespace common::topk {

struct TrafficStats {
	double bytes_moved = 0.0;
	double compare_ops = 0.0;
	bool bytes_exact = false;
	bool ops_exact = false;
};

struct BasicRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
	double end_to_end_stdev_ms = 0.0;
	double algorithm_stdev_ms = 0.0;
	double end_to_end_min_ms = 0.0;
	double algorithm_min_ms = 0.0;
	common::energy::Summary energy;
	TrafficStats traffic;
};

struct BitonicRunStats {
	std::size_t layer_count = 0;
	std::size_t full_comparators = 0;
	std::size_t trunc_comparators = 0;
	const BasicRunStats* full_run_stats = nullptr;
	const BasicRunStats* trunc_run_stats = nullptr;
};

struct MapReduceRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
	double end_to_end_stdev_ms = 0.0;
	double algorithm_stdev_ms = 0.0;
	double end_to_end_min_ms = 0.0;
	double algorithm_min_ms = 0.0;
	std::size_t tiles_used = 0;
	std::size_t aggregated_candidates = 0;
	common::energy::Summary energy;
	TrafficStats traffic;
};

struct GroundTruthRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
	double end_to_end_stdev_ms = 0.0;
	double algorithm_stdev_ms = 0.0;
	double end_to_end_min_ms = 0.0;
	double algorithm_min_ms = 0.0;
	common::energy::Summary energy;
	TrafficStats traffic;
};

template <typename StatsT, typename ResultT> void fill_timing_stats(StatsT& stats, const ResultT& best) {
	stats.end_to_end_ms = best.e2e.mean;
	stats.algorithm_ms = best.algo.mean;
	stats.end_to_end_stdev_ms = best.e2e.stdev;
	stats.algorithm_stdev_ms = best.algo.stdev;
	stats.end_to_end_min_ms = best.e2e.min;
	stats.algorithm_min_ms = best.algo.min;
	stats.energy = best.energy;
}

} // namespace common::topk
