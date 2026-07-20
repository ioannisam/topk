#include <exception>
#include <iostream>

#include "../include/roofline.hpp"
#include "common/roofline.hpp"

int main(int argc, char** argv) {
	try {
		const common::roofline::Config cfg = common::roofline::parse_args(argc, argv);
		return cpu::roofline::execute(cfg);
	} catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
