#include "../include/algorithm.hpp"
#include "simd_traits.hpp"
#include "cpu_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

namespace cpu::bitonic {

namespace {

struct IntraOp {
	std::size_t stride;
	std::size_t stage;
};

// o & ~(stride - 1) == (o / stride) * stride (which block are we in?)
// o & (stride - 1) == o % stride (which element inside that block?)
// return 2 * base + offset;
inline std::size_t trunc_source_index(std::size_t o, std::size_t stride) {
	return 2 * (o & ~(stride - 1)) + (o & (stride - 1));
}

template <typename Tr, int J> inline void cx_step(typename Tr::Vec& v, std::size_t idx, std::size_t stage) {
	auto s = Tr::template permutex<J>(v);
	v = Tr::blend(Tr::template get_blend_mask<J>(idx, stage), Tr::min(v, s), Tr::max(v, s));
}

template <typename Tr>
inline void apply_step(typename Tr::Vec& v, std::size_t idx, std::size_t stride, std::size_t stage) {
	switch (stride) {
	case 1:
		cx_step<Tr, 1>(v, idx, stage);
		break;
	case 2:
		cx_step<Tr, 2>(v, idx, stage);
		break;
	case 4:
		cx_step<Tr, 4>(v, idx, stage);
		break;
	case 8:
		cx_step<Tr, 8>(v, idx, stage);
		break;
	default:
		break;
	}
}

// loop unrolling
template <typename Tr>
inline void apply_step4(
	typename Tr::Vec& v0,
	typename Tr::Vec& v1,
	typename Tr::Vec& v2,
	typename Tr::Vec& v3,
	std::size_t i,
	std::size_t W,
	std::size_t stride,
	std::size_t stage
) {
	switch (stride) {
	case 1:
		cx_step<Tr, 1>(v0, i, stage);
		cx_step<Tr, 1>(v1, i + W, stage);
		cx_step<Tr, 1>(v2, i + 2 * W, stage);
		cx_step<Tr, 1>(v3, i + 3 * W, stage);
		break;
	case 2:
		cx_step<Tr, 2>(v0, i, stage);
		cx_step<Tr, 2>(v1, i + W, stage);
		cx_step<Tr, 2>(v2, i + 2 * W, stage);
		cx_step<Tr, 2>(v3, i + 3 * W, stage);
		break;
	case 4:
		cx_step<Tr, 4>(v0, i, stage);
		cx_step<Tr, 4>(v1, i + W, stage);
		cx_step<Tr, 4>(v2, i + 2 * W, stage);
		cx_step<Tr, 4>(v3, i + 3 * W, stage);
		break;
	case 8:
		cx_step<Tr, 8>(v0, i, stage);
		cx_step<Tr, 8>(v1, i + W, stage);
		cx_step<Tr, 8>(v2, i + 2 * W, stage);
		cx_step<Tr, 8>(v3, i + 3 * W, stage);
		break;
	default:
		break;
	}
}

template <typename Tr>
inline typename Tr::Vec apply_intra_ops(typename Tr::Vec v, std::size_t idx, const IntraOp* ops, std::size_t nops) {
	for (std::size_t o = 0; o < nops; o++) {
		apply_step<Tr>(v, idx, ops[o].stride, ops[o].stage);
	}
	return v;
}

template <typename T>
inline void replay_intra_scalar(T* ptr, std::size_t begin, std::size_t end, const IntraOp* ops, std::size_t nops) {
	for (std::size_t o = 0; o < nops; o++) {
		const std::size_t stride = ops[o].stride;
		const std::size_t stage = ops[o].stage;
		for (std::size_t t = begin; t < end; t++) {
			const std::size_t ixj = t ^ stride;
			if (ixj <= t) {
				continue;
			}
			const bool ascending = (t & stage) == 0;
			if (ascending) {
				if (ptr[t] > ptr[ixj]) {
					std::swap(ptr[t], ptr[ixj]);
				}
			} else {
				if (ptr[t] < ptr[ixj]) {
					std::swap(ptr[t], ptr[ixj]);
				}
			}
		}
	}
}

// elements fit in a single SIMD register, can be fused into a single kernel
template <typename T, template <typename> class Traits>
void run_fused_intra(T* ptr, std::size_t begin, std::size_t end, const IntraOp* ops, std::size_t nops) {
	using Tr = Traits<T>;
	constexpr std::size_t W = Tr::width;
	const std::size_t vec_end = begin + ((end - begin) / W) * W;
	const std::size_t vec4_end = begin + ((end - begin) / (4 * W)) * (4 * W);

	std::size_t i = begin;
	for (; i < vec4_end; i += 4 * W) {
		auto v0 = Tr::load(ptr + i);
		auto v1 = Tr::load(ptr + i + W);
		auto v2 = Tr::load(ptr + i + 2 * W);
		auto v3 = Tr::load(ptr + i + 3 * W);
		for (std::size_t o = 0; o < nops; o++) {
			apply_step4<Tr>(v0, v1, v2, v3, i, W, ops[o].stride, ops[o].stage);
		}
		Tr::store(ptr + i, v0);
		Tr::store(ptr + i + W, v1);
		Tr::store(ptr + i + 2 * W, v2);
		Tr::store(ptr + i + 3 * W, v3);
	}
	for (; i < vec_end; i += W) {
		auto v = Tr::load(ptr + i);
		v = apply_intra_ops<Tr>(v, i, ops, nops);
		Tr::store(ptr + i, v);
	}

	replay_intra_scalar<T>(ptr, vec_end, end, ops, nops);
}

template <typename T>
bool try_run_fused_intra(
	T* ptr, std::size_t begin, std::size_t end, const IntraOp* ops, std::size_t nops, bool use_avx512
) {
#if defined(__x86_64__) || defined(__i386__)
	if (use_avx512) {
		if constexpr (cpu::simd::has_simd512_width<T>::value) {
			run_fused_intra<T, cpu::simd::SimdTraits512>(ptr, begin, end, ops, nops);
			return true;
		}
	}
	if (cpu::simd::cpu_supports_avx2()) {
		if constexpr (cpu::simd::has_simd256_width<T>::value) {
			run_fused_intra<T, cpu::simd::SimdTraits256>(ptr, begin, end, ops, nops);
			return true;
		}
	}
#else
	(void)ptr;
	(void)begin;
	(void)end;
	(void)ops;
	(void)nops;
	(void)use_avx512;
#endif
	return false;
}

// Normal, no fusion possible
template <typename T, template <typename> class Traits>
void run_layer_inter_simd(
	T* ptr, std::size_t begin, std::size_t end, std::size_t stage, std::size_t stride, std::size_t n
) {
	using TraitsT = Traits<T>;
	std::size_t i = begin;

	while (i < end) {
		if ((i & stride) != 0) {
			i = (i | ((stride << 1) - 1)) + 1;
			continue;
		}

		std::size_t chunk_end = std::min((i | (stride - 1)) + 1, end);
		if (chunk_end > n) {
			chunk_end = n;
		}

		const bool asc = (i & stage) == 0;
		if (asc) {
			for (; i + TraitsT::width - 1 < chunk_end; i += TraitsT::width) {
				std::size_t ixj = i + stride;
				auto v1 = TraitsT::load(ptr + i);
				auto v2 = TraitsT::load(ptr + ixj);

				auto lo = TraitsT::min(v1, v2);
				auto hi = TraitsT::max(v1, v2);

				TraitsT::store(ptr + i, lo);
				TraitsT::store(ptr + ixj, hi);
			}
		} else {
			for (; i + TraitsT::width - 1 < chunk_end; i += TraitsT::width) {
				std::size_t ixj = i + stride;
				auto v1 = TraitsT::load(ptr + i);
				auto v2 = TraitsT::load(ptr + ixj);

				auto lo = TraitsT::min(v1, v2);
				auto hi = TraitsT::max(v1, v2);

				// Store inverted for descending
				TraitsT::store(ptr + i, hi);
				TraitsT::store(ptr + ixj, lo);
			}
		}

		// scalar fallback
		for (; i < chunk_end; i++) {
			const std::size_t ixj = i + stride;
			const bool ascending = (i & stage) == 0;
			if (ascending) {
				if (ptr[i] > ptr[ixj]) {
					std::swap(ptr[i], ptr[ixj]);
				}
			} else {
				if (ptr[i] < ptr[ixj]) {
					std::swap(ptr[i], ptr[ixj]);
				}
			}
		}
	}
}

template <typename T>
bool try_run_inter(
	T* ptr, std::size_t begin, std::size_t end, std::size_t stage, std::size_t stride, std::size_t n, bool use_avx512
) {
#if defined(__x86_64__) || defined(__i386__)
	if (use_avx512) {
		if constexpr (cpu::simd::has_simd512_width<T>::value) {
			if (stride >= cpu::simd::SimdTraits512<T>::width) {
				run_layer_inter_simd<T, cpu::simd::SimdTraits512>(ptr, begin, end, stage, stride, n);
				return true;
			}
		}
	}
	if (cpu::simd::cpu_supports_avx2()) {
		if constexpr (cpu::simd::has_simd256_width<T>::value) {
			if (stride >= cpu::simd::SimdTraits256<T>::width) {
				run_layer_inter_simd<T, cpu::simd::SimdTraits256>(ptr, begin, end, stage, stride, n);
				return true;
			}
		}
	}
#else
	(void)ptr;
	(void)begin;
	(void)end;
	(void)stage;
	(void)stride;
	(void)n;
	(void)use_avx512;
#endif
	return false;
}

template <typename T>
void run_normal_scalar(
	T* ptr, std::size_t begin, std::size_t end, std::size_t stage, std::size_t stride, std::size_t active_n
) {
	std::size_t i = begin;
	while (i < end) {
		if ((i & stride) != 0) {
			i = (i | ((stride << 1) - 1)) + 1;
			continue;
		}
		std::size_t chunk_end = std::min((i | (stride - 1)) + 1, end);
		if (chunk_end > active_n) {
			chunk_end = active_n;
		}
		for (; i < chunk_end; i++) {
			const std::size_t ixj = i + stride;
			const bool ascending = (i & stage) == 0;
			if (ascending) {
				if (ptr[i] > ptr[ixj]) {
					std::swap(ptr[i], ptr[ixj]);
				}
			} else {
				if (ptr[i] < ptr[ixj]) {
					std::swap(ptr[i], ptr[ixj]);
				}
			}
		}
	}
}

constexpr std::size_t kTileCapBytes = 524288;
constexpr std::size_t kTileMinBytes = 16384;

template <typename T> std::size_t pow2_floor_elems(std::size_t bytes) {
	std::size_t w = bytes / sizeof(T);
	std::size_t p = 1;
	while (p * 2 <= w)
		p <<= 1;
	return p < 16 ? 16 : p;
}

template <typename T> std::size_t tile_cap_elems() {
	return pow2_floor_elems<T>(kTileCapBytes);
}

template <typename T> std::size_t run_tile_elems(std::size_t max_stride) {
	std::size_t want = 16;
	while (want < (max_stride << 1))
		want <<= 1;
	const std::size_t lo = pow2_floor_elems<T>(kTileMinBytes);
	return std::min(tile_cap_elems<T>(), std::max(lo, want));
}

// elements fit in a cache, can be fused into a single kernel
template <typename T>
void run_tiled(
	T* ptr,
	std::size_t begin,
	std::size_t end,
	const IntraOp* ops,
	std::size_t nops,
	std::size_t active_n,
	std::size_t tile_w,
	std::size_t width,
	bool use_avx512
) {
	for (std::size_t tb = begin; tb < end; tb += tile_w) {
		const std::size_t te = std::min(tb + tile_w, end);
		for (std::size_t o = 0; o < nops; o++) {
			const std::size_t stride = ops[o].stride;
			const std::size_t stage = ops[o].stage;
			if (stride < width) {
				const IntraOp one{stride, stage};
				if (!try_run_fused_intra<T>(ptr, tb, te, &one, 1, use_avx512)) {
					run_normal_scalar(ptr, tb, te, stage, stride, active_n);
				}
			} else {
				if (!try_run_inter<T>(ptr, tb, te, stage, stride, active_n, use_avx512)) {
					run_normal_scalar(ptr, tb, te, stage, stride, active_n);
				}
			}
		}
	}
}

// Truncate, no fusion possible
template <typename T, template <typename> class Traits>
void run_layer_truncate_simd(
	const T* src, T* dst, std::size_t begin, std::size_t end, std::size_t stride, std::size_t n
) {
	using TraitsT = Traits<T>;
	std::size_t i = begin;

	// precompute masks to avoid integer division in the hot loop
	const std::size_t stride_minus_1 = stride - 1;
	const std::size_t stride_mask = ~stride_minus_1;

	while (i < end) {
		if ((i & stride) != 0) {
			i = (i | ((stride << 1) - 1)) + 1;
			continue;
		}

		std::size_t chunk_end = std::min((i | stride_minus_1) + 1, end);
		if (chunk_end > n) {
			chunk_end = n;
		}

		for (; i + TraitsT::width - 1 < chunk_end; i += TraitsT::width) {
			std::size_t ixj = i + stride;

			// 1-cycle bitwise calculation instead of slow division
			std::size_t out_idx = ((i >> 1) & stride_mask) | (i & stride_minus_1);

			auto v1 = TraitsT::load(src + i);
			auto v2 = TraitsT::load(src + ixj);
			auto winner = TraitsT::min(v1, v2);

			TraitsT::store(dst + out_idx, winner);
		}

		for (; i < chunk_end; i++) {
			std::size_t ixj = i + stride;
			std::size_t out_idx = ((i >> 1) & stride_mask) | (i & stride_minus_1);
			dst[out_idx] = std::min(src[i], src[ixj]);
		}
	}
}

template <typename T>
bool try_run_simd_layer_truncate(
	const T* src, T* dst, std::size_t begin, std::size_t end, std::size_t stride, std::size_t n, bool use_avx512
) {
#if defined(__x86_64__) || defined(__i386__)
	if (use_avx512) {
		if constexpr (cpu::simd::has_simd512_width<T>::value) {
			if (stride >= cpu::simd::SimdTraits512<T>::width) {
				run_layer_truncate_simd<T, cpu::simd::SimdTraits512>(src, dst, begin, end, stride, n);
				return true;
			}
		}
	}
	if (cpu::simd::cpu_supports_avx2()) {
		if constexpr (cpu::simd::has_simd256_width<T>::value) {
			if (stride >= cpu::simd::SimdTraits256<T>::width) {
				run_layer_truncate_simd<T, cpu::simd::SimdTraits256>(src, dst, begin, end, stride, n);
				return true;
			}
		}
	}
#else
	(void)src;
	(void)dst;
	(void)begin;
	(void)end;
	(void)stride;
	(void)n;
	(void)use_avx512;
#endif
	return false;
}

// Truncate + IntraRun, can be fused into a single kernel
template <typename T, template <typename> class Traits>
void run_fused_trunc_resort(
	const T* src, T* dst, std::size_t obegin, std::size_t oend, const IntraOp* ops, std::size_t nops
) {
	using Tr = Traits<T>;
	constexpr std::size_t W = Tr::width;
	const std::size_t ovec_end = obegin + ((oend - obegin) / W) * W;
	const std::size_t ovec4_end = obegin + ((oend - obegin) / (4 * W)) * (4 * W);

	std::size_t o = obegin;
	for (; o < ovec4_end; o += 4 * W) {
		auto v0 = Tr::min(Tr::load(src + 2 * o), Tr::load(src + 2 * o + W));
		auto v1 = Tr::min(Tr::load(src + 2 * (o + W)), Tr::load(src + 2 * (o + W) + W));
		auto v2 = Tr::min(Tr::load(src + 2 * (o + 2 * W)), Tr::load(src + 2 * (o + 2 * W) + W));
		auto v3 = Tr::min(Tr::load(src + 2 * (o + 3 * W)), Tr::load(src + 2 * (o + 3 * W) + W));
		for (std::size_t oi = 0; oi < nops; oi++) {
			apply_step4<Tr>(v0, v1, v2, v3, o, W, ops[oi].stride, ops[oi].stage);
		}
		Tr::store(dst + o, v0);
		Tr::store(dst + o + W, v1);
		Tr::store(dst + o + 2 * W, v2);
		Tr::store(dst + o + 3 * W, v3);
	}
	for (; o < ovec_end; o += W) {
		const std::size_t in_base = 2 * o;
		auto v_lo = Tr::load(src + in_base);
		auto v_hi = Tr::load(src + in_base + W);
		auto v = Tr::min(v_lo, v_hi);
		v = apply_intra_ops<Tr>(v, o, ops, nops);
		Tr::store(dst + o, v);
	}

	for (std::size_t t = ovec_end; t < oend; t++) {
		const std::size_t in_base = trunc_source_index(t, W);
		dst[t] = std::min(src[in_base], src[in_base + W]);
	}
	replay_intra_scalar<T>(dst, ovec_end, oend, ops, nops);
}

template <typename T>
bool try_run_fused_trunc_resort(
	const T* src, T* dst, std::size_t obegin, std::size_t oend, const IntraOp* ops, std::size_t nops, bool use_avx512
) {
#if defined(__x86_64__) || defined(__i386__)
	if (use_avx512) {
		if constexpr (cpu::simd::has_simd512_width<T>::value) {
			run_fused_trunc_resort<T, cpu::simd::SimdTraits512>(src, dst, obegin, oend, ops, nops);
			return true;
		}
	}
	if (cpu::simd::cpu_supports_avx2()) {
		if constexpr (cpu::simd::has_simd256_width<T>::value) {
			run_fused_trunc_resort<T, cpu::simd::SimdTraits256>(src, dst, obegin, oend, ops, nops);
			return true;
		}
	}
#else
	(void)src;
	(void)dst;
	(void)obegin;
	(void)oend;
	(void)ops;
	(void)nops;
	(void)use_avx512;
#endif
	return false;
}

class SpinBarrier {
  public:
	explicit SpinBarrier(std::size_t participants) : threshold(participants), count(participants), generation(0) {
	}

	void wait() {
		const std::size_t gen = generation.load(std::memory_order_acquire);

		if (count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			count.store(threshold, std::memory_order_relaxed);
			generation.store(gen + 1, std::memory_order_release);
		} else {
			while (generation.load(std::memory_order_acquire) == gen) {
#if defined(__x86_64__) || defined(__i386__)
				_mm_pause();
#else
				std::this_thread::yield();
#endif
			}
		}
	}

  private:
	std::size_t threshold;
	alignas(64) std::atomic<std::size_t> count;
	alignas(64) std::atomic<std::size_t> generation;
};

// limit unecessary trips to the RAM
enum class GroupKind {
	IntraRun,
	TiledRun,
	InterNormal,
	Truncate,
	TruncResort,
};

struct Group {
	GroupKind kind;
	std::size_t active_n;
	std::size_t out_active_n;
	std::size_t stride;
	std::size_t stage;
	std::size_t ops_begin;
	std::size_t ops_count;
};

template <typename T>
std::vector<Group> build_groups(
	const std::vector<common::bitonic::Layer>& layers, std::vector<IntraOp>& ops, bool use_avx512, bool use_avx2
) {
	const std::size_t width = cpu::simd::simd_block_width<T>(use_avx512, use_avx2);
	const std::size_t imax = width > 1 ? width / 2 : 0;
	const std::size_t tile_thresh = width > 1 ? tile_cap_elems<T>() / 2 : 0;
	std::vector<Group> groups;
	groups.reserve(layers.size());

	bool run_open = false;
	std::size_t run_active_n = 0;
	std::size_t run_ops_begin = 0;
	std::size_t run_max_stride = 0;

	auto flush_run = [&]() {
		if (!run_open) {
			return;
		}
		const GroupKind kind = run_max_stride < width ? GroupKind::IntraRun : GroupKind::TiledRun;
		groups.push_back(Group{kind, run_active_n, 0, run_max_stride, 0, run_ops_begin, ops.size() - run_ops_begin});
		run_open = false;
	};

	for (std::size_t li = 0; li < layers.size(); li++) {
		const common::bitonic::Layer& layer = layers[li];

		if (layer.type == common::bitonic::LayerType::Truncate) {
			flush_run();

			if (width > 0 && layer.stride == width) {
				const std::size_t out_active_n = layer.active_n / 2;
				const std::size_t ops_begin = ops.size();

				// look ahead to see if we can fuse the truncation with the next few sorts
				std::size_t look = li + 1;
				while (look < layers.size() && layers[look].type == common::bitonic::LayerType::Normal &&
					   layers[look].active_n == out_active_n && layers[look].stride <= imax) {
					ops.push_back(IntraOp{layers[look].stride, layers[look].stage});
					look++;
				}
				groups.push_back(
					Group{
						GroupKind::TruncResort,
						layer.active_n,
						out_active_n,
						layer.stride,
						layer.stage,
						ops_begin,
						ops.size() - ops_begin
					}
				);
				li = look - 1; // skip the resort layers we just absorbed
				continue;
			}

			groups.push_back(Group{GroupKind::Truncate, layer.active_n, 0, layer.stride, layer.stage, 0, 0});
			continue;
		}

		// nearby elemets' kernels fused
		if (tile_thresh > 0 && layer.stride <= tile_thresh) {
			if (run_open && run_active_n != layer.active_n) {
				flush_run();
			}
			if (!run_open) {
				run_open = true;
				run_active_n = layer.active_n;
				run_ops_begin = ops.size();
				run_max_stride = 0;
			}
			ops.push_back(IntraOp{layer.stride, layer.stage});
			run_max_stride = std::max(run_max_stride, layer.stride);
		} else {
			flush_run(); // close previous group
			groups.push_back(Group{GroupKind::InterNormal, layer.active_n, 0, layer.stride, layer.stage, 0, 0});
		}
	}
	flush_run();
	return groups;
}

template <typename T> double group_traffic_bytes(const std::vector<Group>& groups) {
	double elems = 0.0;
	for (const auto& g : groups) {
		const double active = static_cast<double>(g.active_n);
		switch (g.kind) {
		case GroupKind::IntraRun:
		case GroupKind::TiledRun:
		case GroupKind::InterNormal:
			elems += 2.0 * active;
			break;
		case GroupKind::Truncate:
			elems += active + active / 2.0;
			break;
		case GroupKind::TruncResort:
			elems += active + static_cast<double>(g.out_active_n);
			break;
		}
	}
	return elems * static_cast<double>(sizeof(T));
}

} // namespace

template <typename T>
void run_topk(
	std::vector<T>& data,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t workers,
	double* out_bytes,
	std::size_t* out_workers
) {
	const std::size_t n = data.size();

	workers = std::min<std::size_t>(workers, cpu::kMaxWorkers);
	workers = std::min(workers, std::max<std::size_t>(1, n >> 16));

	if (out_workers != nullptr) {
		*out_workers = workers;
	}

	std::size_t max_layer_stage = 0;
	for (const auto& layer : layers) {
		max_layer_stage = std::max(max_layer_stage, layer.stage);
	}
	const bool use_avx512 = cpu::simd::use_avx512() &&
							max_layer_stage <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
	const bool use_avx2 = cpu::simd::cpu_supports_avx2();

	std::vector<IntraOp> ops;
	const std::vector<Group> groups = build_groups<T>(layers, ops, use_avx512, use_avx2);

	if (out_bytes != nullptr) {
		*out_bytes = group_traffic_bytes<T>(groups);
	}

	bool needs_alt = false;
	for (const auto& g : groups) {
		if (g.kind == GroupKind::Truncate || g.kind == GroupKind::TruncResort) {
			needs_alt = true;
			break;
		}
	}
	static std::vector<T> alt_storage;
	if (needs_alt && alt_storage.size() < n) {
		alt_storage.resize(n);
	}
	T* src = data.data();
	T* dst = needs_alt ? alt_storage.data() : nullptr;

	SpinBarrier barrier(workers);
	static cpu::utils::WorkerPool pool(workers - 1);

	auto worker_fn = [&](std::size_t tid) {
		for (const auto& group : groups) {
			const std::size_t active_n = group.active_n;
			const bool writes_dst = group.kind == GroupKind::Truncate || group.kind == GroupKind::TruncResort;
			const std::size_t slice_n = group.kind == GroupKind::TruncResort ? group.out_active_n : active_n;

			// typical cache line size is 64 bytes
			const std::size_t gran =
				group.kind == GroupKind::TiledRun ? run_tile_elems<T>(group.stride) : (64 / sizeof(T));

			std::size_t effective_workers = workers;
			if (slice_n < effective_workers * gran) {
				effective_workers = std::max<std::size_t>(1, slice_n / gran);
			}

			std::size_t begin = 0;
			std::size_t end = 0;
			if (tid < effective_workers) {
				std::size_t raw_begin = (slice_n * tid) / effective_workers;
				std::size_t raw_end = (slice_n * (tid + 1)) / effective_workers;

				begin = (raw_begin / gran) * gran;
				end = (tid + 1 == effective_workers) ? slice_n : ((raw_end / gran) * gran);
			}

			if (begin >= end) {
				barrier.wait();
				if (writes_dst) {
					if (tid == 0) {
						std::swap(src, dst);
					}
					barrier.wait();
				}
				continue;
			}

			switch (group.kind) {
			case GroupKind::IntraRun: {
				if (!try_run_fused_intra<T>(
						src, begin, end, ops.data() + group.ops_begin, group.ops_count, use_avx512
					)) {
					for (std::size_t o = 0; o < group.ops_count; o++) {
						const IntraOp& op = ops[group.ops_begin + o];
						run_normal_scalar(src, begin, end, op.stage, op.stride, active_n);
					}
				}
				break;
			}
			case GroupKind::TiledRun: {
				run_tiled<T>(
					src,
					begin,
					end,
					ops.data() + group.ops_begin,
					group.ops_count,
					active_n,
					gran,
					cpu::simd::simd_block_width<T>(use_avx512, use_avx2),
					use_avx512
				);
				break;
			}
			case GroupKind::InterNormal: {
				if (!try_run_inter<T>(src, begin, end, group.stage, group.stride, active_n, use_avx512)) {
					run_normal_scalar(src, begin, end, group.stage, group.stride, active_n);
				}
				break;
			}
			case GroupKind::Truncate: {
				if (!try_run_simd_layer_truncate<T>(src, dst, begin, end, group.stride, active_n, use_avx512)) {
					std::size_t i = begin;
					const std::size_t stride_minus_1 = group.stride - 1;
					const std::size_t stride_mask = ~stride_minus_1;
					while (i < end) {
						if ((i & group.stride) != 0) {
							i = (i | ((group.stride << 1) - 1)) + 1;
							continue;
						}
						std::size_t chunk_end = std::min((i | stride_minus_1) + 1, end);
						if (chunk_end > active_n) {
							chunk_end = active_n;
						}
						for (; i < chunk_end; i++) {
							const std::size_t ixj = i + group.stride;
							const std::size_t out_idx = ((i >> 1) & stride_mask) | (i & stride_minus_1);
							dst[out_idx] = std::min(src[i], src[ixj]);
						}
					}
				}
				break;
			}
			case GroupKind::TruncResort: {
				if (!try_run_fused_trunc_resort<T>(
						src, dst, begin, end, ops.data() + group.ops_begin, group.ops_count, use_avx512
					)) {
					for (std::size_t o = begin; o < end; o++) {
						const std::size_t in_base = trunc_source_index(o, group.stride);
						dst[o] = std::min(src[in_base], src[in_base + group.stride]);
					}
					for (std::size_t oi = 0; oi < group.ops_count; oi++) {
						const IntraOp& op = ops[group.ops_begin + oi];
						run_normal_scalar(dst, begin, end, op.stage, op.stride, group.out_active_n);
					}
				}
				break;
			}
			}

			barrier.wait();

			if (writes_dst) {
				if (tid == 0) {
					std::swap(src, dst);
				}
				barrier.wait();
			}
		}
	};

	pool.dispatch([&](std::size_t tid) {
		if (tid < workers - 1) {
			worker_fn(tid);
		}
	});
	worker_fn(workers - 1);
	pool.join();

	const std::size_t result_n = layers.empty() ? n : layers.back().active_n;
	if (src != data.data()) {
		std::copy(src, src + result_n, data.begin());
	}
	data.resize(result_n);
}

template void run_topk<std::int32_t>(
	std::vector<std::int32_t>& data,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t workers,
	double* out_bytes,
	std::size_t* out_workers
);
template void run_topk<std::uint32_t>(
	std::vector<std::uint32_t>& data,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t workers,
	double* out_bytes,
	std::size_t* out_workers
);
template void run_topk<float>(
	std::vector<float>& data,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t workers,
	double* out_bytes,
	std::size_t* out_workers
);
template void run_topk<double>(
	std::vector<double>& data,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t workers,
	double* out_bytes,
	std::size_t* out_workers
);
#if defined(__FLT16_MANT_DIG__)
template void run_topk<_Float16>(
	std::vector<_Float16>& data,
	const std::vector<common::bitonic::Layer>& layers,
	std::size_t workers,
	double* out_bytes,
	std::size_t* out_workers
);
#endif

} // namespace cpu::bitonic
