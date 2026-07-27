#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace gpu::utils {

#define CUDA_CHECK(expr)                                                                                               \
	do {                                                                                                               \
		cudaError_t _err = (expr);                                                                                     \
		if (_err != cudaSuccess) {                                                                                     \
			throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));                          \
		}                                                                                                              \
	} while (false)

template <typename T> struct DeviceBuffer {
	T* ptr = nullptr;
	std::size_t size = 0;

	explicit DeviceBuffer(std::size_t num_elements) : size(num_elements) {
		if (size > 0) {
			CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
		}
	}

	~DeviceBuffer() {
		if (ptr != nullptr) {
			cudaFree(ptr);
			ptr = nullptr;
		}
	}

	DeviceBuffer(const DeviceBuffer&) = delete;
	DeviceBuffer& operator=(const DeviceBuffer&) = delete;

	DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), size(other.size) {
		other.ptr = nullptr;
		other.size = 0;
	}

	DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
		if (this != &other) {
			if (ptr != nullptr) {
				cudaFree(ptr);
			}
			ptr = other.ptr;
			size = other.size;
			other.ptr = nullptr;
			other.size = 0;
		}
		return *this;
	}

	T* get() const {
		return ptr;
	}
	T* operator->() const {
		return ptr;
	}
	T& operator[](std::size_t idx) const {
		return ptr[idx];
	}
};

} // namespace gpu::utils
