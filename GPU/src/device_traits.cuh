#pragma once

#include <cstdint>
#include <cfloat>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <type_traits>

namespace gpu::traits {

template <typename T> struct DeviceType {
	using type = T;
};

#if defined(__FLT16_MANT_DIG__)
template <> struct DeviceType<_Float16> {
	using type = __half;
};
#endif

template <typename T> struct DeviceTraits;

template <> struct DeviceTraits<float> {
	using Vec2 = float2;
	static __device__ __forceinline__ bool gt(float a, float b) {
		return a > b;
	}
	static __device__ __forceinline__ bool lt(float a, float b) {
		return a < b;
	}
	static __device__ __forceinline__ float min(float a, float b) {
		return fminf(a, b);
	}
	static __device__ __forceinline__ float max(float a, float b) {
		return fmaxf(a, b);
	}
	static __device__ __forceinline__ float sentinel(bool want_max) {
		return want_max ? -CUDART_INF_F : CUDART_INF_F;
	}
};

template <> struct DeviceTraits<double> {
	using Vec2 = double2;
	static __device__ __forceinline__ bool gt(double a, double b) {
		return a > b;
	}
	static __device__ __forceinline__ bool lt(double a, double b) {
		return a < b;
	}
	static __device__ __forceinline__ double min(double a, double b) {
		return fmin(a, b);
	}
	static __device__ __forceinline__ double max(double a, double b) {
		return fmax(a, b);
	}
	static __device__ __forceinline__ double sentinel(bool want_max) {
		return want_max ? -CUDART_INF : CUDART_INF;
	}
};

template <> struct DeviceTraits<__half> {
	using Vec2 = __half2;
	static __device__ __forceinline__ bool gt(__half a, __half b) {
		return __hgt(a, b);
	}
	static __device__ __forceinline__ bool lt(__half a, __half b) {
		return __hlt(a, b);
	}
	static __device__ __forceinline__ __half min(__half a, __half b) {
		return __hmin(a, b);
	}
	static __device__ __forceinline__ __half max(__half a, __half b) {
		return __hmax(a, b);
	}
	static __device__ __forceinline__ __half sentinel(bool want_max) {
		return want_max ? __float2half(-65504.0f) : __float2half(65504.0f);
	}
};

template <> struct DeviceTraits<int32_t> {
	using Vec2 = int2;
	static __device__ __forceinline__ bool gt(int32_t a, int32_t b) {
		return a > b;
	}
	static __device__ __forceinline__ bool lt(int32_t a, int32_t b) {
		return a < b;
	}
	static __device__ __forceinline__ int32_t min(int32_t a, int32_t b) {
		return a < b ? a : b;
	}
	static __device__ __forceinline__ int32_t max(int32_t a, int32_t b) {
		return a > b ? a : b;
	}
	static __device__ __forceinline__ int32_t sentinel(bool want_max) {
		return want_max ? INT32_MIN : INT32_MAX;
	}
};

template <> struct DeviceTraits<uint32_t> {
	using Vec2 = uint2;
	static __device__ __forceinline__ bool gt(uint32_t a, uint32_t b) {
		return a > b;
	}
	static __device__ __forceinline__ bool lt(uint32_t a, uint32_t b) {
		return a < b;
	}
	static __device__ __forceinline__ uint32_t min(uint32_t a, uint32_t b) {
		return a < b ? a : b;
	}
	static __device__ __forceinline__ uint32_t max(uint32_t a, uint32_t b) {
		return a > b ? a : b;
	}
	static __device__ __forceinline__ uint32_t sentinel(bool want_max) {
		return want_max ? 0 : UINT32_MAX;
	}
};

} // namespace gpu::traits
