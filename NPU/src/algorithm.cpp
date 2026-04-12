#include "../include/algorithm.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <optional>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

namespace npu::bitonic {

namespace {

template <typename T>
std::vector<std::size_t> build_offsets(const std::vector<std::vector<unsigned char>>& keep,
									   const std::vector<common::bitonic::Layer>& layers,
									   std::vector<std::uint32_t>& pairs_out, std::size_t n) {
	std::vector<std::size_t> offsets;
	offsets.reserve(layers.size() + 1);
	offsets.push_back(0);

	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const std::size_t j = layers[layer_idx].j;
		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t ixj = i ^ j;
			if (ixj <= i) {
				continue;
			}
			if (keep[layer_idx][i] || keep[layer_idx][ixj]) {
				pairs_out.push_back(static_cast<std::uint32_t>(i));
			}
		}
		offsets.push_back(pairs_out.size());
	}

	return offsets;
}

xrt::device open_device() {
	try {
		return xrt::device{0};
	} catch (const std::exception& ex) {
		throw std::runtime_error(std::string("Failed to open NPU device 0 via XRT: ") + ex.what());
	}
}

struct OffloadConfig {
	bool enabled = false;
	std::string xclbin_path;
	std::string kernel_name = "bitonic_layer";
};

bool xclbin_uses_dpu_abi(const std::string& xclbin_path) {
	const std::size_t slash = xclbin_path.find_last_of('/');
	const std::string dir = (slash == std::string::npos) ? std::string(".") : xclbin_path.substr(0, slash);
	const std::string metadata = dir + "/embedded_metadata.xml";

	std::ifstream in(metadata);
	if (!in.good()) {
		return false;
	}

	std::string xml((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	return xml.find("type=\"dpu\"") != std::string::npos && xml.find("name=\"opcode\"") != std::string::npos &&
		   xml.find("name=\"ninstr\"") != std::string::npos;
}

const char* read_env(const char* key) {
	const char* value = std::getenv(key);
	if (value == nullptr || value[0] == '\0') {
		return nullptr;
	}
	return value;
}

OffloadConfig load_offload_config() {
	OffloadConfig cfg;
	if (const char* xclbin = read_env("NPU_OFFLOAD_XCLBIN")) {
		cfg.enabled = true;
		cfg.xclbin_path = xclbin;
	}
	if (const char* kernel = read_env("NPU_OFFLOAD_KERNEL")) {
		cfg.kernel_name = kernel;
	}
	return cfg;
}

std::size_t safe_group_id(const xrt::kernel& kernel, int arg_index) {
	try {
		return static_cast<std::size_t>(kernel.group_id(arg_index));
	} catch (...) {
		return 0;
	}
}

std::uint64_t read_opcode() {
	if (const char* op = read_env("NPU_OFFLOAD_OPCODE")) {
		return static_cast<std::uint64_t>(std::stoull(op));
	}
	return 3;
}

bool ninstr_is_bytes() {
	if (const char* mode = read_env("NPU_OFFLOAD_NINSTR_BYTES")) {
		const std::string token(mode);
		return token == "1" || token == "true" || token == "TRUE" || token == "on";
	}
	return false;
}

unsigned int read_wait_timeout_ms() {
	if (const char* timeout = read_env("NPU_OFFLOAD_WAIT_MS")) {
		return static_cast<unsigned int>(std::stoul(timeout));
	}
	return 5000u;
}

const char* cmd_state_name(ert_cmd_state state) {
	switch (state) {
	case ERT_CMD_STATE_NEW:
		return "NEW";
	case ERT_CMD_STATE_QUEUED:
		return "QUEUED";
	case ERT_CMD_STATE_RUNNING:
		return "RUNNING";
	case ERT_CMD_STATE_COMPLETED:
		return "COMPLETED";
	case ERT_CMD_STATE_ERROR:
		return "ERROR";
	case ERT_CMD_STATE_ABORT:
		return "ABORT";
	case ERT_CMD_STATE_SUBMITTED:
		return "SUBMITTED";
	case ERT_CMD_STATE_TIMEOUT:
		return "TIMEOUT";
	case ERT_CMD_STATE_NORESPONSE:
		return "NORESPONSE";
	case ERT_CMD_STATE_SKERROR:
		return "SKERROR";
	case ERT_CMD_STATE_SKCRASHED:
		return "SKCRASHED";
	default:
		return "UNKNOWN";
	}
}

void wait_for_run_or_throw(xrt::run& run, unsigned int timeout_ms, const char* launch_kind) {
	const ert_cmd_state state = run.wait(timeout_ms);
	if (state == ERT_CMD_STATE_COMPLETED) {
		return;
	}

	if (state == ERT_CMD_STATE_TIMEOUT) {
		(void)run.abort();
	}

	std::ostringstream oss;
	oss << "NPU " << launch_kind << " command failed: state=" << cmd_state_name(state) << " ("
		<< static_cast<int>(state) << ")"
		<< ", wait_ms=" << timeout_ms;
	throw std::runtime_error(oss.str());
}

void wait_for_runlist_or_throw(const xrt::runlist& rl, unsigned int timeout_ms, const char* launch_kind) {
	if (rl.wait(std::chrono::milliseconds(timeout_ms)) == std::cv_status::timeout) {
		std::ostringstream oss;
		oss << "NPU " << launch_kind << " command failed: state=TIMEOUT"
			<< ", wait_ms=" << timeout_ms;
		throw std::runtime_error(oss.str());
	}
}

std::vector<std::uint32_t> read_dpu_instr_words(const std::string& xclbin_path, const std::string& kernel_name) {
	if (const char* instr_override = read_env("NPU_OFFLOAD_INSTR")) {
		std::ifstream in(instr_override, std::ios::binary);
		if (in.good()) {
			std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
			if (!bytes.empty()) {
				const std::size_t padded = ((bytes.size() + 3u) / 4u) * 4u;
				bytes.resize(padded, 0);
				std::vector<std::uint32_t> words(padded / 4u, 0u);
				std::memcpy(words.data(), bytes.data(), padded);
				return words;
			}
		}
	}

	const std::size_t slash = xclbin_path.find_last_of('/');
	const std::string dir = (slash == std::string::npos) ? std::string(".") : xclbin_path.substr(0, slash);
	const std::string instr_file = dir + "/" + kernel_name + "_ctrlpkt_dma_main.bin";

	std::ifstream in(instr_file, std::ios::binary);
	if (!in.good()) {
		return {};
	}

	std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	if (bytes.empty()) {
		return {};
	}

	const std::size_t padded = ((bytes.size() + 3u) / 4u) * 4u;
	bytes.resize(padded, 0);
	std::vector<std::uint32_t> words(padded / 4u, 0u);
	std::memcpy(words.data(), bytes.data(), padded);
	return words;
}

xrt::bo alloc_bo_for_kernel(const std::optional<xrt::hw_context>& hwctx, const xrt::device& dev, size_t bytes,
							std::size_t group_id, bool dpu_abi) {
	(void)hwctx;
	(void)dpu_abi;
	return xrt::bo(dev, bytes, xrt::bo::flags::host_only, group_id);
}

template <typename T>
RunStats run_network_offload_xrt(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
								 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
								 const OffloadConfig& offload_cfg) {
	if constexpr (sizeof(T) != 4) {
		throw std::runtime_error("NPU offload currently supports only 4-byte element types (int/uint/float)");
	}

	const std::size_t n = data.size();
	const std::size_t full_pairs = n >> 1;
	std::size_t active_comparators = full_pairs * layers.size();

	std::vector<std::uint32_t> pair_i;
	std::vector<std::size_t> offsets;
	if (trunc) {
		offsets = build_offsets<T>(keep, layers, pair_i, n);
		active_comparators = pair_i.size();
	}

	xrt::device dev = open_device();
	std::optional<xrt::hw_context> hwctx;
	xrt::uuid uuid;
	xrt::kernel kernel;
	xrt::xclbin xclbin(offload_cfg.xclbin_path);
	uuid = dev.register_xclbin(xclbin);
	hwctx.emplace(dev, uuid);
	kernel = xrt::kernel(*hwctx, offload_cfg.kernel_name);

	const bool dpu_abi = xclbin_uses_dpu_abi(offload_cfg.xclbin_path);
	const std::uint64_t opcode = read_opcode();
	const unsigned int wait_timeout_ms = read_wait_timeout_ms();

	const std::size_t data_bytes = n * sizeof(T);
	const std::size_t data_group = dpu_abi ? safe_group_id(kernel, 3) : safe_group_id(kernel, 0);
	xrt::bo data_bo = alloc_bo_for_kernel(hwctx, dev, data_bytes, data_group, dpu_abi);
	std::memcpy(data_bo.map<void*>(), data.data(), data_bytes);
	data_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	const std::size_t pair_count = pair_i.empty() ? 1 : pair_i.size();
	const std::size_t pairs_group = dpu_abi ? safe_group_id(kernel, 4) : safe_group_id(kernel, 1);
	xrt::bo pairs_bo = alloc_bo_for_kernel(hwctx, dev, pair_count * sizeof(std::uint32_t), pairs_group, dpu_abi);
	if (!pair_i.empty()) {
		std::memcpy(pairs_bo.map<void*>(), pair_i.data(), pair_i.size() * sizeof(std::uint32_t));
	}
	pairs_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	xrt::bo instr_bo;
	xrt::bo bo2;
	xrt::bo bo3;
	xrt::bo bo4;
	std::uint32_t ninstr = 0;
	if (dpu_abi) {
		const std::vector<std::uint32_t> instr_words =
			read_dpu_instr_words(offload_cfg.xclbin_path, offload_cfg.kernel_name);
		ninstr = static_cast<std::uint32_t>(instr_words.size());
		const std::size_t instr_bytes =
			std::max<std::size_t>(sizeof(std::uint32_t), instr_words.size() * sizeof(std::uint32_t));
		if (ninstr_is_bytes()) {
			ninstr = static_cast<std::uint32_t>(instr_bytes);
		}

		const std::size_t instr_group = safe_group_id(kernel, 1);
		const std::size_t bo2_group = safe_group_id(kernel, 5);
		const std::size_t bo3_group = safe_group_id(kernel, 6);
		const std::size_t bo4_group = safe_group_id(kernel, 7);
		instr_bo = xrt::bo(dev, instr_bytes, xrt::bo::flags::cacheable, instr_group);
		bo2 = xrt::bo(dev, 1u, xrt::bo::flags::host_only, bo2_group);
		bo3 = xrt::bo(dev, 8u, xrt::bo::flags::host_only, bo3_group);
		bo4 = xrt::bo(dev, 1u, xrt::bo::flags::host_only, bo4_group);
		std::uint32_t* inst = instr_bo.map<std::uint32_t*>();
		if (!instr_words.empty()) {
			std::memcpy(inst, instr_words.data(), instr_words.size() * sizeof(std::uint32_t));
		} else {
			inst[0] = 0;
		}
		std::memset(bo2.map<void*>(), 0, 1u);
		std::memset(bo3.map<void*>(), 0, 8u);
		std::memset(bo4.map<void*>(), 0, 1u);
		instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		bo2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		bo3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
		bo4.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	}

	std::size_t launches = 0;

	auto t0 = std::chrono::high_resolution_clock::now();
	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const std::uint32_t stage = static_cast<std::uint32_t>(layers[layer_idx].k);
		const std::uint32_t step = static_cast<std::uint32_t>(layers[layer_idx].j);

		std::uint32_t begin = 0;
		std::uint32_t count = static_cast<std::uint32_t>(full_pairs);
		if (trunc) {
			begin = static_cast<std::uint32_t>(offsets[layer_idx]);
			count = static_cast<std::uint32_t>(offsets[layer_idx + 1] - offsets[layer_idx]);
			if (count == 0) {
				continue;
			}
		}

		if (!dpu_abi) {
			auto run = kernel(data_bo, pairs_bo, static_cast<std::uint32_t>(n), stage, step, begin, count,
							  static_cast<std::uint32_t>(trunc ? 1 : 0));
			wait_for_run_or_throw(run, wait_timeout_ms, "offload");
		} else {
			xrt::run run(kernel);
			run.set_arg(0, static_cast<std::uint64_t>(opcode));
			run.set_arg(1, instr_bo);
			run.set_arg(2, ninstr);
			run.set_arg(3, data_bo);
			run.set_arg(4, pairs_bo);
			run.set_arg(5, bo2);
			run.set_arg(6, bo3);
			run.set_arg(7, bo4);
			xrt::runlist rl(*hwctx);
			rl.add(std::move(run));
			rl.execute();
			wait_for_runlist_or_throw(rl, wait_timeout_ms, "dpu");
		}
		launches++;
	}
	auto t1 = std::chrono::high_resolution_clock::now();

	data_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	std::memcpy(data.data(), data_bo.map<void*>(), data_bytes);

	const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
	if (dpu_abi && launches == 0) {
		throw std::runtime_error("Kernel image reports DPU ABI and produced no launchable work.");
	}
	return RunStats{elapsed_ms, launches, active_comparators, 1, true};
}
} // namespace

std::string query_device_name() {
	xrt::device dev = open_device();
	return dev.get_info<xrt::info::device::name>();
}

std::string query_device_bdf() {
	xrt::device dev = open_device();
	return dev.get_info<xrt::info::device::bdf>();
}

bool is_offload_configured() {
	return load_offload_config().enabled;
}

template <typename T>
RunStats run_network_npu(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers,
						 const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers) {
	(void)workers;

	if (data.empty()) {
		return RunStats{0.0, 0, 0, 1, true};
	}

	// Validate runtime access to the NPU device for this backend invocation.
	(void)open_device();

	const OffloadConfig offload_cfg = load_offload_config();
	if (!offload_cfg.enabled) {
		throw std::runtime_error(
			"NPU offload is required for this backend. Set NPU_OFFLOAD_XCLBIN to a valid xclbin path.");
	}

	return run_network_offload_xrt(data, layers, keep, trunc, offload_cfg);
}

template RunStats run_network_npu<std::int32_t>(std::vector<std::int32_t>& data,
												const std::vector<common::bitonic::Layer>& layers,
												const std::vector<std::vector<unsigned char>>& keep, bool trunc,
												std::size_t workers);
template RunStats run_network_npu<std::uint32_t>(std::vector<std::uint32_t>& data,
												 const std::vector<common::bitonic::Layer>& layers,
												 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
												 std::size_t workers);
template RunStats run_network_npu<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
										 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
										 std::size_t workers);
template RunStats run_network_npu<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers,
										  const std::vector<std::vector<unsigned char>>& keep, bool trunc,
										  std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template RunStats run_network_npu<_Float16>(std::vector<_Float16>& data,
											const std::vector<common::bitonic::Layer>& layers,
											const std::vector<std::vector<unsigned char>>& keep, bool trunc,
											std::size_t workers);
#endif

} // namespace npu::bitonic
