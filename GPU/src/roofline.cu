#include "../include/roofline.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/benchmark.hpp"
#include "common/random.hpp"

namespace gpu::roofline {

namespace {

using common::roofline::Config;
using common::roofline::Experiment;
using common::roofline::Point;

#define CUDA_CHECK(expr)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (expr);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));                          \
		}                                                                                                              \
	} while (false)

constexpr int kBlockSize = 256;
constexpr int kChains = 4;
constexpr int kSettleSeconds = 3;

__global__ void fma_kernel(const float* __restrict__ src, float* __restrict__ out, std::size_t n, int ops) {
	const float a = 1.0000001f;
	const float b = 0.0000001f;

	float acc = 0.0f;
	const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x * kChains;
	std::size_t base = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kChains;

	for (; base + kChains <= n; base += stride) {
		float v[kChains];
#pragma unroll
		for (int c = 0; c < kChains; ++c) {
			v[c] = src[base + c];
		}
		for (int j = 0; j < ops; ++j) {
#pragma unroll
			for (int c = 0; c < kChains; ++c) {
				v[c] = v[c] * a + b;
			}
		}
#pragma unroll
		for (int c = 0; c < kChains; ++c) {
			acc += v[c];
		}
	}

	out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void copy_kernel(const float* __restrict__ src, float* __restrict__ dst, std::size_t n) {
	const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
	for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
		dst[i] = src[i];
	}
}

int resolve_grid_size(std::size_t n) {
	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	const int by_occupancy = prop.multiProcessorCount * 32;
	const std::size_t by_work =
		(n + static_cast<std::size_t>(kBlockSize) * kChains - 1) / (static_cast<std::size_t>(kBlockSize) * kChains);
	return std::max(1, static_cast<int>(std::min<std::size_t>(by_occupancy, std::max<std::size_t>(1, by_work))));
}

template <typename Fn>
Point measure(const char* kernel, int ops, std::size_t n, double bytes_moved, double flops, Fn&& fn) {
	auto best = common::benchmark::run_benchmark([&]() -> common::benchmark::TimedValue<double> {
		const auto t0 = std::chrono::high_resolution_clock::now();
		common::energy::FullScope energy_scope;

		fn();
		CUDA_CHECK(cudaDeviceSynchronize());

		energy_scope.close();
		const auto t1 = std::chrono::high_resolution_clock::now();
		const double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
		return common::benchmark::TimedValue<double>{elapsed, elapsed, 0.0};
	});

	Point point;
	point.kernel = kernel;
	point.ops_per_elem = ops;
	point.elements = n;
	point.bytes_moved = bytes_moved;
	point.flops = flops;
	point.ms_mean = best.e2e.mean;
	point.ms_stdev = best.e2e.stdev;
	point.ms_min = best.e2e.min;
	point.energy = best.energy;
	return point;
}

} // namespace

int execute(const Config& cfg) {
	const std::size_t n = std::max<std::size_t>(kBlockSize * kChains, cfg.bytes / sizeof(float));
	const double read_bytes = static_cast<double>(n) * sizeof(float);
	const int grid_size = resolve_grid_size(n);

	const std::vector<float> host_src = common::utils::generate_random_input<float>(n, cfg.seed, 0, 1000);

	float* d_src = nullptr;
	float* d_dst = nullptr;
	float* d_out = nullptr;
	CUDA_CHECK(cudaMalloc(&d_src, n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(grid_size) * kBlockSize * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(d_src, host_src.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	const auto settle_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(kSettleSeconds);
	while (std::chrono::steady_clock::now() < settle_deadline) {
		for (int i = 0; i < 16; ++i) {
			fma_kernel<<<grid_size, kBlockSize>>>(d_src, d_out, n, 8);
		}
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	std::vector<Point> points;

	if (cfg.experiment == Experiment::Stream || cfg.experiment == Experiment::Both) {
		points.push_back(measure("read", 0, n, read_bytes, static_cast<double>(n),
								 [&]() { fma_kernel<<<grid_size, kBlockSize>>>(d_src, d_out, n, 0); }));
		points.push_back(measure("copy", 0, n, 2.0 * read_bytes, 0.0,
								 [&]() { copy_kernel<<<grid_size, kBlockSize>>>(d_src, d_dst, n); }));
	}

	if (cfg.experiment == Experiment::Sweep || cfg.experiment == Experiment::Both) {
		for (const int ops : cfg.ops) {
			const double flops = static_cast<double>(n) * (2.0 * static_cast<double>(ops) + 1.0);
			points.push_back(measure("fma", ops, n, read_bytes, flops,
									 [&]() { fma_kernel<<<grid_size, kBlockSize>>>(d_src, d_out, n, ops); }));
		}
	}

	CUDA_CHECK(cudaFree(d_src));
	CUDA_CHECK(cudaFree(d_dst));
	CUDA_CHECK(cudaFree(d_out));

	common::roofline::report(cfg, "gpu", points);
	return 0;
}

} // namespace gpu::roofline
