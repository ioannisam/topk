#include "../include/roofline.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/benchmark.hpp"
#include "common/random.hpp"
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
	for (std::size_t i = 0; i < n; ++i) {
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

	if (cfg.experiment == Experiment::Stream || cfg.experiment == Experiment::Both) {
		points.push_back(
			measure("read", 0, n, bytes, static_cast<double>(n), [&]() { return read_buffer(src_map, n); }));
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

	common::roofline::report(cfg, "npu", points);
	return 0;
}

} // namespace npu::roofline
