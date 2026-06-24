#include "../include/algorithm.hpp"
#include "device_traits.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>

namespace gpu::ground_truth {

namespace {

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
		if (ptr) {
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
			if (ptr)
				cudaFree(ptr);
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

} // namespace

template <typename T> double run_topk(T* data, std::size_t n, std::size_t k, bool want_max) {
	using D = typename gpu::traits::DeviceType<T>::type;

	if (k == 0 || n == 0) {
		return 0.0;
	}

	DeviceBuffer<D> d_data(n);

	CUDA_CHECK(cudaMemcpy(d_data.get(), data, n * sizeof(D), cudaMemcpyHostToDevice));

	thrust::device_ptr<D> dev_ptr(d_data.get());

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	CUDA_CHECK(cudaEventRecord(start));

	if (want_max) {
		thrust::sort(thrust::device, dev_ptr, dev_ptr + n, thrust::greater<D>());
	} else {
		thrust::sort(thrust::device, dev_ptr, dev_ptr + n, thrust::less<D>());
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));

	float algo_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&algo_ms, start, stop));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	std::size_t out_size = std::min(k, n);

	CUDA_CHECK(cudaMemcpy(data, d_data.get(), out_size * sizeof(D), cudaMemcpyDeviceToHost));

	return static_cast<double>(algo_ms);
}

// Explicit instantiations
template double run_topk<std::int32_t>(std::int32_t* data, std::size_t n, std::size_t k, bool want_max);
template double run_topk<std::uint32_t>(std::uint32_t* data, std::size_t n, std::size_t k, bool want_max);
template double run_topk<float>(float* data, std::size_t n, std::size_t k, bool want_max);
template double run_topk<double>(double* data, std::size_t n, std::size_t k, bool want_max);
#if defined(__FLT16_MANT_DIG__)
template double run_topk<_Float16>(_Float16* data, std::size_t n, std::size_t k, bool want_max);
#endif

} // namespace gpu::ground_truth
