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

	// forbid copying
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

template <typename T> struct PinnedBuffer {
	T* ptr = nullptr;
	std::size_t size = 0;

	explicit PinnedBuffer(std::size_t num_elements) : size(num_elements) {
		if (size > 0) {
			CUDA_CHECK(cudaMallocHost(&ptr, size * sizeof(T)));
		}
	}

	~PinnedBuffer() {
		if (ptr != nullptr) {
			cudaFreeHost(ptr);
			ptr = nullptr;
		}
	}

	// forbid copying
	PinnedBuffer(const PinnedBuffer&) = delete;
	PinnedBuffer& operator=(const PinnedBuffer&) = delete;

	PinnedBuffer(PinnedBuffer&& other) noexcept : ptr(other.ptr), size(other.size) {
		other.ptr = nullptr;
		other.size = 0;
	}

	PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
		if (this != &other) {
			if (ptr != nullptr) {
				cudaFreeHost(ptr);
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
};

struct StreamGuard {
	cudaStream_t handle = nullptr;

	StreamGuard() {
		CUDA_CHECK(cudaStreamCreate(&handle));
	}
	~StreamGuard() {
		if (handle != nullptr) {
			cudaStreamDestroy(handle);
		}
	}
	StreamGuard(const StreamGuard&) = delete;
	StreamGuard& operator=(const StreamGuard&) = delete;

	cudaStream_t get() const {
		return handle;
	}
};

struct EventGuard {
	cudaEvent_t handle = nullptr;

	EventGuard() {
		CUDA_CHECK(cudaEventCreate(&handle));
	}
	~EventGuard() {
		if (handle != nullptr) {
			cudaEventDestroy(handle);
		}
	}
	EventGuard(const EventGuard&) = delete;
	EventGuard& operator=(const EventGuard&) = delete;

	cudaEvent_t get() const {
		return handle;
	}
};

struct GraphGuard {
	cudaGraph_t handle = nullptr;

	GraphGuard() = default;
	~GraphGuard() {
		if (handle != nullptr) {
			cudaGraphDestroy(handle);
		}
	}
	GraphGuard(const GraphGuard&) = delete;
	GraphGuard& operator=(const GraphGuard&) = delete;

	cudaGraph_t* addr() {
		return &handle;
	}
	cudaGraph_t get() const {
		return handle;
	}
};

struct GraphExecGuard {
	cudaGraphExec_t handle = nullptr;

	GraphExecGuard() = default;
	~GraphExecGuard() {
		if (handle != nullptr) {
			cudaGraphExecDestroy(handle);
		}
	}
	GraphExecGuard(const GraphExecGuard&) = delete;
	GraphExecGuard& operator=(const GraphExecGuard&) = delete;

	cudaGraphExec_t* addr() {
		return &handle;
	}
	cudaGraphExec_t get() const {
		return handle;
	}
};

} // namespace gpu::utils
