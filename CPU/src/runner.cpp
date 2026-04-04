#include "runner.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <thread>
#include <vector>

#include "algorithm.hpp"
#include "layers.hpp"
#include "reporting.hpp"
#include "utils.hpp"

namespace {

constexpr int randMin = 0;
constexpr int randMax = 1000;

struct Context {
	std::size_t n;
	std::size_t hw_threads;
	std::size_t ex_threads;
};

struct Check {
	bool ok;
	double full_ms;
};

Context build_context(const Config& cfg) {
	const std::size_t n = std::size_t{1} << cfg.q;
	const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
	std::size_t ex_threads = cfg.ex_threads == 0 ? hw_threads : cfg.ex_threads;
	ex_threads = std::max<std::size_t>(1, std::min(ex_threads, n));

	return Context{n, hw_threads, ex_threads};
}

void apply_mode_transform(std::vector<int>& data, bool want_max) {
	if (!want_max) {
		return;
	}

	for (std::size_t i = 0; i < data.size(); i++) {
		data[i] = -data[i];
	}
}

double run_truncated_network(std::vector<int>& truncated, const std::vector<Layer>& layers,
							 const std::vector<std::vector<unsigned char>>& keep, std::size_t ex_threads) {

	auto t0 = std::chrono::high_resolution_clock::now();
	run_network_parallel(truncated, layers, keep, true, ex_threads);
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

Check run_check(const Config& cfg, const std::vector<int>& input, const std::vector<int>& truncated,
				const std::vector<Layer>& layers, const std::vector<std::vector<unsigned char>>& keep,
				std::size_t ex_threads) {

	if (!cfg.run_check) {
		return Check{true, 0.0};
	}

	std::vector<int> full = input;
	auto t0 = std::chrono::high_resolution_clock::now();
	run_network_parallel(full, layers, keep, false, ex_threads);
	auto t1 = std::chrono::high_resolution_clock::now();
	const double full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

	for (std::size_t i = 0; i < cfg.k; i++) {
		if (truncated[i] != full[i]) {
			return Check{false, full_ms};
		}
	}

	return Check{true, full_ms};
}

std::vector<int> build_output(const std::vector<int>& truncated, const Config& cfg) {
	std::vector<int> output(cfg.k);
	for (std::size_t i = 0; i < cfg.k; i++) {
		output[i] = cfg.want_max ? -truncated[i] : truncated[i];
	}

	if (cfg.sort_output) {
		if (cfg.want_max) {
			std::sort(output.begin(), output.end(), std::greater<int>());
		} else {
			std::sort(output.begin(), output.end());
		}
	}

	return output;
}

} // namespace

int run_topk(const Config& cfg) {
	const Context ctx = build_context(cfg);

	print_configuration(cfg, ctx.ex_threads, ctx.n);
	auto layers = build_layers(ctx.n);
	auto keep = build_masks(layers, ctx.n, cfg.k);

	const std::size_t full_cmp = count_full_comparators(layers, ctx.n);
	const std::size_t trunc_cmp = count_truncated_comparators(layers, keep, ctx.n);

	std::vector<int> input = generate_random_input(ctx.n, cfg.seed, randMin, randMax);
	apply_mode_transform(input, cfg.want_max);

	std::vector<int> truncated = input;
	const double trunc_ms = run_truncated_network(truncated, layers, keep, ctx.ex_threads);
	const Check check = run_check(cfg, input, truncated, layers, keep, ctx.ex_threads);

	print_timing_summary(cfg.run_check, check.full_ms, trunc_ms);
	print_debug_metrics(cfg, ctx.hw_threads, ctx.ex_threads, layers.size(), full_cmp, trunc_cmp,
						check.full_ms, trunc_ms);

	if (cfg.run_check) {
		print_correctness_summary(cfg.run_check, check.ok);
		if (!check.ok) {
			return 2;
		}
	}

	std::vector<int> output = build_output(truncated, cfg);
	print_topk_output(cfg, output);

	return 0;
}
