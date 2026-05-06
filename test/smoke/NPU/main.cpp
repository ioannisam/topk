#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>

#include <xrt/xrt_device.h>
#include <xrt/experimental/xrt_xclbin.h>

int main() {
	try {
		xrt::device dev{0};
		std::cout << "NPU device found: " << dev.get_info<xrt::info::device::name>() << "\n";

		const char* xclbin_path = std::getenv("NPU_OFFLOAD_XCLBIN");
		if (!xclbin_path) {
			std::cerr << "NPU smoke test: FAIL (NPU_OFFLOAD_XCLBIN not set)\n";
			return 1;
		}

		// Construct the string cleanly first
		std::string path_str(xclbin_path);

		// Use a distinct variable name to avoid namespace collisions
		xrt::xclbin kernel_bin(path_str);
		auto uuid = dev.register_xclbin(kernel_bin);

		std::cout << "Successfully loaded xclbin: " << xclbin_path << "\n";
		std::cout << "NPU smoke test: PASS\n";
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "NPU smoke test: FAIL\n";
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}
}
