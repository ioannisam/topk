#pragma once

#include "common/energy.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

namespace npu::utils {

inline xrt::device open_device() {
	try {
		return xrt::device{0};
	} catch (const std::exception& ex) {
		throw std::runtime_error(std::string("Failed to open NPU device 0 via XRT: ") + ex.what());
	}
}

struct OffloadConfig {
	bool enabled = false;
	std::string xclbin_path;
	std::string kernel_name = "MLIR_AIE";
};

inline const char* read_env(const char* key) {
	const char* value = std::getenv(key);
	return (value == nullptr || value[0] == '\0') ? nullptr : value;
}

inline OffloadConfig load_offload_config() {
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

inline std::string xclbin_stem(const std::string& xclbin_path) {
	return std::filesystem::path(xclbin_path).stem().string();
}

inline void require_xclbin_for(const OffloadConfig& cfg, const char* algorithm) {
	if (!cfg.enabled) {
		return;
	}
	const std::string stem = xclbin_stem(cfg.xclbin_path);
	if (stem == algorithm) {
		return;
	}
	throw std::runtime_error(
		"NPU_OFFLOAD_XCLBIN is " + cfg.xclbin_path + " but algo=" + algorithm + " needs the " + algorithm +
		" xclbin. Point NPU_OFFLOAD_XCLBIN at " + algorithm + ".xclbin (with " + algorithm +
		".bin beside it), or run a matching algo."
	);
}

inline std::size_t safe_group_id(const xrt::kernel& kernel, int arg_index) {
	if (arg_index < 3) {
		return kernel.group_id(arg_index);
	}
	return 0;
}

inline unsigned int read_wait_timeout_ms() {
	if (const char* timeout = read_env("NPU_OFFLOAD_WAIT_MS")) {
		return static_cast<unsigned int>(std::stoul(timeout));
	}
	return 5000u;
}

inline void wait_for_runlist_or_throw(const xrt::runlist& rl, unsigned int timeout_ms) {
	common::energy::Scope energy_scope(common::energy::Channel::Wait);
	const auto status = rl.wait(std::chrono::milliseconds(timeout_ms));
	energy_scope.close();
	if (status == std::cv_status::timeout) {
		throw std::runtime_error("NPU command timed out");
	}
}

inline std::vector<uint32_t> load_instruction_sequence(const std::string& xclbin_path) {
	const std::filesystem::path xclbin_fs(xclbin_path);
	const std::filesystem::path dir = xclbin_fs.has_parent_path() ? xclbin_fs.parent_path() : ".";
	const std::string path = (dir / (xclbin_stem(xclbin_path) + ".bin")).string();

	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		throw std::runtime_error("Could not find NPU instruction sequence at: " + path);
	}

	std::streamsize size = file.tellg();
	if (size <= 0) {
		throw std::runtime_error("Instruction file is empty (0 bytes): " + path);
	}
	if (static_cast<std::size_t>(size) % sizeof(uint32_t) != 0) {
		throw std::runtime_error("Instruction file is not a whole number of 32-bit words: " + path);
	}
	file.seekg(0, std::ios::beg);

	std::vector<uint32_t> insts(size / sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(insts.data()), size);

	return insts;
}

struct SharedXrtState {
	xrt::device dev;
	xrt::xclbin xclbin;
	xrt::uuid uuid;
	xrt::hw_context hwctx;
	xrt::kernel kernel;
	std::vector<uint32_t> instr_v;
	xrt::bo instr_bo;

	std::vector<xrt::bo> mr_cfg_bo;
	std::vector<xrt::bo> mr_dst_bo;
	std::vector<xrt::bo> mr_src_bo;
	std::size_t mr_batch_bytes = 0;

	std::vector<xrt::bo> bit_dst_bo;
	std::vector<xrt::bo> bit_src_bo;
	std::size_t bit_batch_bytes = 0;

	SharedXrtState(const OffloadConfig& cfg) {
		dev = open_device();
		xclbin = xrt::xclbin(cfg.xclbin_path);
		uuid = dev.register_xclbin(xclbin);
		hwctx = xrt::hw_context(dev, uuid);
		kernel = xrt::kernel(hwctx, cfg.kernel_name);

		const std::size_t instr_group = safe_group_id(kernel, 1);
		instr_v = load_instruction_sequence(cfg.xclbin_path);
		instr_bo = xrt::bo(dev, instr_v.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, instr_group);
		std::memcpy(instr_bo.map<void*>(), instr_v.data(), instr_v.size() * sizeof(uint32_t));
		instr_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	}

	void allocate_mr_buffers(std::size_t batch_bytes) {
		if (mr_batch_bytes >= batch_bytes) {
			return;
		}

		mr_cfg_bo.clear();
		mr_dst_bo.clear();
		mr_src_bo.clear();

		for (int i = 0; i < 2; i++) {
			mr_cfg_bo.push_back(
				xrt::bo(dev, 4096 * sizeof(int32_t), xrt::bo::flags::host_only, safe_group_id(kernel, 3))
			);
			mr_dst_bo.push_back(xrt::bo(dev, batch_bytes, xrt::bo::flags::host_only, safe_group_id(kernel, 4)));
			mr_src_bo.push_back(xrt::bo(dev, batch_bytes, xrt::bo::flags::host_only, safe_group_id(kernel, 5)));
		}
		mr_batch_bytes = batch_bytes;
	}

	void allocate_bit_buffers(std::size_t batch_bytes) {
		if (bit_batch_bytes >= batch_bytes) {
			return;
		}

		bit_dst_bo.clear();
		bit_src_bo.clear();

		for (int i = 0; i < 2; i++) {
			bit_dst_bo.push_back(xrt::bo(dev, batch_bytes, xrt::bo::flags::host_only, safe_group_id(kernel, 3)));
			bit_src_bo.push_back(xrt::bo(dev, batch_bytes, xrt::bo::flags::host_only, safe_group_id(kernel, 4)));
		}
		bit_batch_bytes = batch_bytes;
	}
};

inline SharedXrtState& get_shared_xrt_state(const OffloadConfig& offload_cfg) {
	static std::unique_ptr<SharedXrtState> instance = std::make_unique<SharedXrtState>(offload_cfg);
	return *instance;
}

} // namespace npu::utils
