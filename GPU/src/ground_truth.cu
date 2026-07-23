#include "../include/algorithm.hpp"
#include "device_traits.cuh"

#include "common/energy.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>

#include <cub/device/device_topk.cuh>
#include <cuda/execution.determinism.h>
#include <cuda/execution.output_ordering.h>
#include <cuda/execution.require.h>
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

	const std::size_t kk = std::min(k, n);

	auto env = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
										cuda::execution::output_ordering::unsorted);

	DeviceBuffer<D> d_in(n);
	DeviceBuffer<D> d_out(kk);

	CUDA_CHECK(cudaMemcpy(d_in.get(), data, n * sizeof(D), cudaMemcpyHostToDevice));

	std::size_t temp_bytes = 0;
	if (want_max) {
		CUDA_CHECK(cub::DeviceTopK::MaxKeys(nullptr, temp_bytes, d_in.get(), d_out.get(), n, kk, env));
	} else {
		CUDA_CHECK(cub::DeviceTopK::MinKeys(nullptr, temp_bytes, d_in.get(), d_out.get(), n, kk, env));
	}
	DeviceBuffer<std::uint8_t> d_temp(temp_bytes);

	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	common::energy::Scope energy_scope(common::energy::Channel::Algo);
	CUDA_CHECK(cudaEventRecord(start));

	if (want_max) {
		CUDA_CHECK(cub::DeviceTopK::MaxKeys(d_temp.get(), temp_bytes, d_in.get(), d_out.get(), n, kk, env));
	} else {
		CUDA_CHECK(cub::DeviceTopK::MinKeys(d_temp.get(), temp_bytes, d_in.get(), d_out.get(), n, kk, env));
	}

	CUDA_CHECK(cudaEventRecord(stop));
	CUDA_CHECK(cudaEventSynchronize(stop));
	energy_scope.close();

	float algo_ms = 0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&algo_ms, start, stop));

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	CUDA_CHECK(cudaMemcpy(data, d_out.get(), kk * sizeof(D), cudaMemcpyDeviceToHost));

	if constexpr (sizeof(T) == 2) {
		__half* h = reinterpret_cast<__half*>(data);
		if (want_max) {
			std::sort(h, h + kk, [](__half a, __half b) { return __half2float(a) > __half2float(b); });
		} else {
			std::sort(h, h + kk, [](__half a, __half b) { return __half2float(a) < __half2float(b); });
		}
	} else {
		if (want_max) {
			std::sort(data, data + kk, std::greater<T>());
		} else {
			std::sort(data, data + kk, std::less<T>());
		}
	}

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
