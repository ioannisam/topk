#pragma once

#include <cstddef>

namespace common::topk {

struct BasicRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
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
	std::size_t tiles_used = 0;
	std::size_t aggregated_candidates = 0;
};

struct GroundTruthRunStats {
	double end_to_end_ms = 0.0;
	double algorithm_ms = 0.0;
};

} // namespace common::topk
