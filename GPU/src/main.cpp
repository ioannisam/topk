#include <exception>
#include <iostream>

#include "common/config.hpp"
#include "../include/runner.hpp"

int main(int argc, char** argv) {
	try {
		const common::config::Config cfg = common::config::parse_args(argc, argv);
		return gpu::topk::execute(cfg);
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
