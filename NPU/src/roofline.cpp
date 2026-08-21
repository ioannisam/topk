#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "common/benchmark.hpp"
#include "common/random.hpp"
#include "common/roofline.hpp"
#include "xrt_utils.hpp"

namespace npu::roofline {

namespace {

using common::roofline::Config;
using common::roofline::Experiment;
using common::roofline::measure;
using common::roofline::Point;

using Elem = std::int32_t;

__attribute__((noinline)) double read_buffer(const Elem* src, std::size_t n) {
	std::int64_t acc = 0;
	for (std::size_t i = 0; i < n; i++) {
		acc += src[i];
	}
	asm volatile("" : "+r"(acc) : : "memory");
	return static_cast<double>(acc);
}

__attribute__((noinline)) double copy_buffer(const Elem* src, Elem* dst, std::size_t n) {
	std::memcpy(dst, src, n * sizeof(Elem));
	asm volatile("" : : : "memory");
	return static_cast<double>(dst[0]) + static_cast<double>(dst[n - 1]);
}

constexpr std::size_t kCmpLane = 64;

__attribute__((noinline)) double cmp_buffer(const Elem* src, std::size_t n, int ops) {
	Elem x[kCmpLane];
	Elem y[kCmpLane];
	Elem z[kCmpLane];
	std::int64_t acc = 0;
	for (std::size_t l = 0; l < kCmpLane; l++) {
		z[l] = static_cast<Elem>(l);
	}

	std::size_t i = 0;
	for (; i + 2 * kCmpLane <= n; i += 2 * kCmpLane) {
		for (std::size_t l = 0; l < kCmpLane; l++) {
			x[l] = src[i + l];
			y[l] = src[i + kCmpLane + l];
		}
		for (int j = 0; j < ops; j++) {
			for (std::size_t l = 0; l < kCmpLane; l++) {
				const Elem lo = std::min(x[l], y[l]);
				const Elem hi = std::max(x[l], y[l]);
				x[l] = lo;
				y[l] = z[l];
				z[l] = hi;
			}
		}
		for (std::size_t l = 0; l < kCmpLane; l++) {
			acc += x[l] + y[l];
		}
	}

	for (std::size_t l = 0; l < kCmpLane; l++) {
		acc += z[l];
	}
	asm volatile("" : "+r"(acc) : : "memory");
	return static_cast<double>(acc);
}

} // namespace

int execute(const Config& cfg) {
	const npu::utils::OffloadConfig offload_cfg = npu::utils::load_offload_config();
	if (!offload_cfg.enabled) {
		throw std::runtime_error("NPU roofline requires NPU_OFFLOAD_XCLBIN to be set");
	}

	npu::utils::SharedXrtState& state = npu::utils::get_shared_xrt_state(offload_cfg);

	const std::size_t n = std::max<std::size_t>(1024, cfg.bytes / sizeof(Elem));
	const double bytes = static_cast<double>(n) * sizeof(Elem);

	xrt::bo src_bo(state.dev, n * sizeof(Elem), xrt::bo::flags::host_only, npu::utils::safe_group_id(state.kernel, 5));
	xrt::bo dst_bo(state.dev, n * sizeof(Elem), xrt::bo::flags::host_only, npu::utils::safe_group_id(state.kernel, 4));

	Elem* src_map = src_bo.map<Elem*>();
	Elem* dst_map = dst_bo.map<Elem*>();

	const std::vector<Elem> host = common::utils::generate_random_input<Elem>(n, cfg.seed, 0, 1000000);
	std::memcpy(src_map, host.data(), n * sizeof(Elem));
	std::memset(dst_map, 0, n * sizeof(Elem));

	std::vector<Point> points;

	if (common::roofline::includes(cfg.experiment, Experiment::Stream)) {
		points.push_back(measure("read", 0, n, bytes, static_cast<double>(n), [&]() {
			return read_buffer(src_map, n);
		}));
		points.push_back(measure("copy", 0, n, 2.0 * bytes, 0.0, [&]() { return copy_buffer(src_map, dst_map, n); }));
		points.push_back(measure("stage_h2d", 0, n, bytes, 0.0, [&]() {
			src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
			return 0.0;
		}));
		points.push_back(measure("stage_d2h", 0, n, bytes, 0.0, [&]() {
			dst_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
			return 0.0;
		}));
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Sweep)) {
		const std::size_t n_cmp_processed = n - (n % (2 * kCmpLane));
		for (const int ops : cfg.ops) {
			const double cmp_ops = common::roofline::compare_exchange_count(n_cmp_processed, ops);
			points.push_back(measure("cmp", ops, n, bytes, cmp_ops, [&]() { return cmp_buffer(src_map, n, ops); }));
		}
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Cache)) {
		for (const std::size_t size : cfg.sizes) {
			const std::size_t elems = std::max<std::size_t>(1024, size / sizeof(Elem));
			if (elems > n) {
				continue;
			}
			const std::size_t repeats = std::max<std::size_t>(1, n / elems);
			const double moved = static_cast<double>(elems) * sizeof(Elem) * static_cast<double>(repeats);
			points.push_back(measure("cache_read", 0, elems, moved, 0.0, [&]() {
				double acc = 0.0;
				for (std::size_t r = 0; r < repeats; r++) {
					acc += read_buffer(src_map, elems);
				}
				return acc;
			}));
		}
	}

	if (common::roofline::includes(cfg.experiment, Experiment::Transfer)) {
		for (const std::size_t size : cfg.sizes) {
			const std::size_t elems = std::max<std::size_t>(1, size / sizeof(Elem));
			if (elems > n) {
				continue;
			}
			const std::size_t chunk_bytes = elems * sizeof(Elem);
			const double moved = static_cast<double>(chunk_bytes);

			points.push_back(measure("stage_write", 0, elems, moved, 0.0, [&]() {
				std::memcpy(src_map, host.data(), chunk_bytes);
				return 0.0;
			}));
			points.push_back(measure("stage_write_sync", 0, elems, moved, 0.0, [&]() {
				std::memcpy(src_map, host.data(), chunk_bytes);
				src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, chunk_bytes, 0);
				return 0.0;
			}));
			points.push_back(measure("sync_only", 0, elems, moved, 0.0, [&]() {
				src_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, chunk_bytes, 0);
				return 0.0;
			}));
		}
	}

	common::roofline::report(cfg, "npu", points);
	return 0;
}

} // namespace npu::roofline

int main(int argc, char** argv) {
	try {
		const common::roofline::Config cfg = common::roofline::parse_args(argc, argv);
		return npu::roofline::execute(cfg);
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
