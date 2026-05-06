#include <iostream>
#include <cuda_runtime.h>

int main() {
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		std::cerr << "GPU smoke test: FAIL (CUDA Error: " << static_cast<int>(error_id) << ")\n";
		return 1;
	}

	if (deviceCount == 0) {
		std::cerr << "GPU smoke test: FAIL (No CUDA devices found)\n";
		return 1;
	}

	std::cout << "GPU smoke test: PASS\n";
	std::cout << "Detected " << deviceCount << " CUDA Capable device(s)\n";
	return 0;
}
