#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "common/energy.hpp"
#include "common/roofline.hpp"
#include "../include/roofline.hpp"

int main(int argc, char** argv) {
	try {
		const char* device_energy = std::getenv("TOPK_ENERGY_DEVICE");
		if (device_energy != nullptr && std::string(device_energy) == "1") {
			common::energy::enable_device_counter();
		}
		const common::roofline::Config cfg = common::roofline::parse_args(argc, argv);
		return gpu::roofline::execute(cfg);
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
