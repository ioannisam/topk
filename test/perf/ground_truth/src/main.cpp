#include <exception>
#include <iostream>

#include "common/config.hpp"
#include "runner.hpp"

int main(int argc, char** argv) {
    try {
        const common::Config cfg = common::parse_args(argc, argv);
        return run_topk(cfg);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
