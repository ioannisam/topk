#pragma once

#include <cstddef>

#include "common/energy.hpp"

namespace common::topk {

struct BasicRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
	double end_to_end_stdev_ms = 0.0;
	double algorithm_stdev_ms = 0.0;
	double end_to_end_min_ms = 0.0;
	double algorithm_min_ms = 0.0;
	common::energy::Summary energy;
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
};

struct GroundTruthRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
	double end_to_end_stdev_ms = 0.0;
	double algorithm_stdev_ms = 0.0;
	double end_to_end_min_ms = 0.0;
	double algorithm_min_ms = 0.0;
	common::energy::Summary energy;
};

} // namespace common::topk
