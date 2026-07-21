#include "../include/roofline.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
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
using common::roofline::measure;
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

__global__ void repeat_read_kernel(const float* __restrict__ src, float* __restrict__ out, std::size_t n, int repeats) {
	float acc = 0.0f;
	const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
	const std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	for (int r = 0; r < repeats; ++r) {
		for (std::size_t i = base; i < n; i += stride) {
			acc += src[i];
		}
	}
	out[base] = acc;
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

	if (common::roofline::includes(cfg.experiment, Experiment::Stream)) {
		points.push_back(measure("read", 0, n, read_bytes, static_cast<double>(n), [&]() {
			fma_kernel<<<grid_size, kBlockSize>>>(d_src, d_out, n, 0);
			CUDA_CHECK(cudaDeviceSynchronize());
			return 0.0;
		}));
		points.push_back(measure("copy", 0, n, 2.0 * read_bytes, 0.0, [&]() {
			copy_kernel<<<grid_size, kBlockSize>>>(d_src, d_dst, n);
			CUDA_CHECK(cudaDeviceSynchronize());
			return 0.0;
		}));
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Sweep)) {
		for (const int ops : cfg.ops) {
			const double flops = static_cast<double>(n) * (2.0 * static_cast<double>(ops) + 1.0);
			points.push_back(measure("fma", ops, n, read_bytes, flops, [&]() {
				fma_kernel<<<grid_size, kBlockSize>>>(d_src, d_out, n, ops);
				CUDA_CHECK(cudaDeviceSynchronize());
				return 0.0;
			}));
		}
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Cache)) {
		for (const std::size_t size : cfg.sizes) {
			const std::size_t elems = std::max<std::size_t>(kBlockSize * kChains, size / sizeof(float));
			if (elems > n) {
				continue;
			}
			const std::size_t repeats = std::max<std::size_t>(1, n / elems);
			const double moved = static_cast<double>(elems) * sizeof(float) * static_cast<double>(repeats);
			points.push_back(measure("cache_read", 0, elems, moved, 0.0, [&]() {
				repeat_read_kernel<<<grid_size, kBlockSize>>>(d_src, d_out, elems, static_cast<int>(repeats));
				CUDA_CHECK(cudaDeviceSynchronize());
				return 0.0;
			}));
		}
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Transfer)) {
		float* pinned = nullptr;
		CUDA_CHECK(cudaMallocHost(&pinned, n * sizeof(float)));
		std::memcpy(pinned, host_src.data(), n * sizeof(float));
		std::vector<float> pageable(host_src);

		for (const std::size_t size : cfg.sizes) {
			const std::size_t elems = std::max<std::size_t>(1, size / sizeof(float));
			if (elems > n) {
				continue;
			}
			const double moved = static_cast<double>(elems) * sizeof(float);

			points.push_back(measure("h2d_pageable", 0, elems, moved, 0.0, [&]() {
				CUDA_CHECK(cudaMemcpy(d_src, pageable.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
				return 0.0;
			}));
			points.push_back(measure("h2d_pinned", 0, elems, moved, 0.0, [&]() {
				CUDA_CHECK(cudaMemcpy(d_src, pinned, elems * sizeof(float), cudaMemcpyHostToDevice));
				return 0.0;
			}));
			points.push_back(measure("d2h_pageable", 0, elems, moved, 0.0, [&]() {
				CUDA_CHECK(cudaMemcpy(pageable.data(), d_src, elems * sizeof(float), cudaMemcpyDeviceToHost));
				return 0.0;
			}));
			points.push_back(measure("d2h_pinned", 0, elems, moved, 0.0, [&]() {
				CUDA_CHECK(cudaMemcpy(pinned, d_src, elems * sizeof(float), cudaMemcpyDeviceToHost));
				return 0.0;
			}));
		}

		points.push_back(measure("launch", 0, 1, 0.0, 0.0, [&]() {
			fma_kernel<<<1, kBlockSize>>>(d_src, d_out, 0, 0);
			CUDA_CHECK(cudaDeviceSynchronize());
			return 0.0;
		}));

		CUDA_CHECK(cudaFreeHost(pinned));
	}

	CUDA_CHECK(cudaFree(d_src));
	CUDA_CHECK(cudaFree(d_dst));
	CUDA_CHECK(cudaFree(d_out));

	common::roofline::report(cfg, "gpu", points);
	return 0;
}

} // namespace gpu::roofline
