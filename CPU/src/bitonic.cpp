#include "../include/algorithm.hpp"
#include "../include/simd_traits.hpp"

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

template <typename T, int J, template <typename> class Traits>
void run_layer_intra_simd(T* ptr, std::size_t begin, std::size_t end, std::size_t k) {
	using TraitsT = Traits<T>;
	const std::size_t j2 = static_cast<std::size_t>(J) * 2;
	std::size_t i = (begin + (j2 - 1)) & ~(j2 - 1);

	for (; i + TraitsT::width - 1 < end; i += TraitsT::width) {
		auto v = TraitsT::load(ptr + i);
		auto swapped = TraitsT::template permutex<J>(v);
		auto lo = TraitsT::min(v, swapped);
		auto hi = TraitsT::max(v, swapped);

		auto mask = TraitsT::template get_blend_mask<J>(i, k);
		auto out = TraitsT::blend(mask, lo, hi);
		TraitsT::store(ptr + i, out);
	}

	// Scalar fallback for remaining elements
	for (; i < end; ++i) {
		const std::size_t ixj = i ^ static_cast<std::size_t>(J);
		if (ixj <= i)
			continue;
		const bool ascending = (i & k) == 0;
		if (ascending) {
			if (ptr[i] > ptr[ixj])
				std::swap(ptr[i], ptr[ixj]);
		} else {
			if (ptr[i] < ptr[ixj])
				std::swap(ptr[i], ptr[ixj]);
		}
	}
}

template <typename T, template <typename> class Traits>
void run_layer_inter_simd(T* ptr, std::size_t begin, std::size_t end, std::size_t k, std::size_t j, std::size_t n) {
	using TraitsT = Traits<T>;
	std::size_t i = begin;

	while (i < end) {
		if ((i & j) != 0) {
			i = (i | ((j << 1) - 1)) + 1;
			continue;
		}

		std::size_t chunk_end = std::min((i | (j - 1)) + 1, end);
		if (chunk_end > n)
			chunk_end = n;

        const bool asc = (i & k) == 0;
        if (asc) {
            for (; i + TraitsT::width - 1 < chunk_end; i += TraitsT::width) {
                std::size_t ixj = i + j;
                auto v1 = TraitsT::load(ptr + i);
                auto v2 = TraitsT::load(ptr + ixj);

                auto lo = TraitsT::min(v1, v2);
                auto hi = TraitsT::max(v1, v2);

                TraitsT::store(ptr + i, lo);
                TraitsT::store(ptr + ixj, hi);
            }
        } else {
            for (; i + TraitsT::width - 1 < chunk_end; i += TraitsT::width) {
                std::size_t ixj = i + j;
                auto v1 = TraitsT::load(ptr + i);
                auto v2 = TraitsT::load(ptr + ixj);

                auto lo = TraitsT::min(v1, v2);
                auto hi = TraitsT::max(v1, v2);

                // Store inverted for descending
                TraitsT::store(ptr + i, hi);
                TraitsT::store(ptr + ixj, lo);
            }
        }

		// Scalar fallback
		for (; i < chunk_end; ++i) {
			const std::size_t ixj = i + j;
			const bool ascending = (i & k) == 0;
			if (ascending) {
				if (ptr[i] > ptr[ixj])
					std::swap(ptr[i], ptr[ixj]);
			} else {
				if (ptr[i] < ptr[ixj])
					std::swap(ptr[i], ptr[ixj]);
			}
		}
	}
}

template <typename T, template <typename> class Traits>
void run_layer_truncate_simd(const T* src, T* dst, std::size_t begin, std::size_t end, std::size_t j, std::size_t n) {
	using TraitsT = Traits<T>;
	std::size_t i = begin;

	// Precompute masks to avoid integer division in the hot loop
	const std::size_t j_minus_1 = j - 1;
	const std::size_t j_mask = ~j_minus_1;

	while (i < end) {
		if ((i & j) != 0) {
			i = (i | ((j << 1) - 1)) + 1;
			continue;
		}

		std::size_t chunk_end = std::min((i | j_minus_1) + 1, end);
		if (chunk_end > n)
			chunk_end = n;

		for (; i + TraitsT::width - 1 < chunk_end; i += TraitsT::width) {
			std::size_t ixj = i + j;

			// 1-cycle bitwise calculation instead of slow division
			std::size_t out_idx = ((i >> 1) & j_mask) | (i & j_minus_1);

			auto v1 = TraitsT::load(src + i);
			auto v2 = TraitsT::load(src + ixj);
			auto winner = TraitsT::min(v1, v2);

			TraitsT::store(dst + out_idx, winner);
		}

		for (; i < chunk_end; ++i) {
			std::size_t ixj = i + j;
			std::size_t out_idx = ((i >> 1) & j_mask) | (i & j_minus_1);
			dst[out_idx] = std::min(src[i], src[ixj]);
		}
	}
}

template <typename T>
bool try_run_simd_layer_normal(T* ptr, std::size_t begin, std::size_t end, std::size_t k, std::size_t j,
							   std::size_t n) {
#if defined(__x86_64__) || defined(__i386__)
	if (cpu::simd::cpu_supports_avx512f() && k <= std::numeric_limits<std::int32_t>::max()) {
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>) {
			if (j <= 8) {
				if (j == 1)
					run_layer_intra_simd<T, 1, cpu::simd::SimdTraits512>(ptr, begin, end, k);
				else if (j == 2)
					run_layer_intra_simd<T, 2, cpu::simd::SimdTraits512>(ptr, begin, end, k);
				else if (j == 4)
					run_layer_intra_simd<T, 4, cpu::simd::SimdTraits512>(ptr, begin, end, k);
				else if (j == 8)
					run_layer_intra_simd<T, 8, cpu::simd::SimdTraits512>(ptr, begin, end, k);
				return true;
			}
			if (j >= 16) {
				run_layer_inter_simd<T, cpu::simd::SimdTraits512>(ptr, begin, end, k, j, n);
				return true;
			}
		}
	}

	if (cpu::simd::cpu_supports_avx2()) {
		// 32-bit types (Width = 8)
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>) {
			if (j <= 4) {
				if (j == 1)
					run_layer_intra_simd<T, 1, cpu::simd::SimdTraits256>(ptr, begin, end, k);
				else if (j == 2)
					run_layer_intra_simd<T, 2, cpu::simd::SimdTraits256>(ptr, begin, end, k);
				else if (j == 4)
					run_layer_intra_simd<T, 4, cpu::simd::SimdTraits256>(ptr, begin, end, k);
				return true;
			}
			if (j >= 8) {
				run_layer_inter_simd<T, cpu::simd::SimdTraits256>(ptr, begin, end, k, j, n);
				return true;
			}
		}

		// 64-bit types (Width = 4)
		if constexpr (std::is_same_v<T, double>) {
			if (j <= 2) {
				if (j == 1)
					run_layer_intra_simd<T, 1, cpu::simd::SimdTraits256>(ptr, begin, end, k);
				else if (j == 2)
					run_layer_intra_simd<T, 2, cpu::simd::SimdTraits256>(ptr, begin, end, k);
				return true;
			}
			if (j >= 4) {
				run_layer_inter_simd<T, cpu::simd::SimdTraits256>(ptr, begin, end, k, j, n);
				return true;
			}
		}
	}
#endif
	return false;
}

template <typename T>
bool try_run_simd_layer_truncate(const T* src, T* dst, std::size_t begin, std::size_t end, std::size_t j,
								 std::size_t n) {
#if defined(__x86_64__) || defined(__i386__)
	if (cpu::simd::cpu_supports_avx512f()) {
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>) {
			if (j >= 16) {
				run_layer_truncate_simd<T, cpu::simd::SimdTraits512>(src, dst, begin, end, j, n);
				return true;
			}
		}
	}
	if (cpu::simd::cpu_supports_avx2()) {
		if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::int32_t> || std::is_same_v<T, std::uint32_t>) {
			if (j >= 8) {
				run_layer_truncate_simd<T, cpu::simd::SimdTraits256>(src, dst, begin, end, j, n);
				return true;
			}
		}
		if constexpr (std::is_same_v<T, double>) {
			if (j >= 4) {
				run_layer_truncate_simd<T, cpu::simd::SimdTraits256>(src, dst, begin, end, j, n);
				return true;
			}
		}
	}
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
	std::atomic<std::size_t> count;
	std::atomic<std::size_t> generation;
};

} // namespace

template <typename T>
void run_topk(std::vector<T>& data, const std::vector<common::bitonic::Layer>& layers, std::size_t workers) {
	const std::size_t n = data.size();

	// Allocate a secondary buffer to ping-pong compacted arrays
	std::vector<T> alt_buffer(n);
	T* src = data.data();
	T* dst = alt_buffer.data();

	SpinBarrier barrier(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers);

	for (std::size_t tid = 0; tid < workers; tid++) {
		pool.emplace_back([&, tid]() {
			for (std::size_t layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
				const auto& layer = layers[layer_idx];
				const std::size_t active_n = layer.active_n;

				std::size_t effective_workers = workers;
				if (active_n < effective_workers * 16) {
					effective_workers = std::max<std::size_t>(1, active_n / 16);
				}

				std::size_t begin = 0;
				std::size_t end = 0;
				if (tid < effective_workers) {
					std::size_t raw_begin = (active_n * tid) / effective_workers;
					std::size_t raw_end = (active_n * (tid + 1)) / effective_workers;

					// snap boundaries to the nearest multiple of 16 for perfect SIMD alignment
					begin = (raw_begin / 16) * 16;
					end = (tid + 1 == effective_workers) ? active_n : ((raw_end / 16) * 16);
				}

				if (begin >= end) {
					barrier.wait();
					if (tid == 0 && layer.type == common::bitonic::LayerType::Truncate) {
						std::swap(src, dst);
					}
					barrier.wait();
					continue;
				}

				if (layer.type == common::bitonic::LayerType::Normal) {
					// try SIMD-accelerated layer
					if (!try_run_simd_layer_normal(src, begin, end, layer.k, layer.j, active_n)) {
						// fallback scalar loop
						std::size_t i = begin;
						while (i < end) {
							if ((i & layer.j) != 0) {
								i = (i | ((layer.j << 1) - 1)) + 1;
								continue;
							}

							std::size_t chunk_end = std::min((i | (layer.j - 1)) + 1, end);
							if (chunk_end > active_n)
								chunk_end = active_n;

							for (; i < chunk_end; ++i) {
								const std::size_t ixj = i + layer.j;
								const bool ascending = (i & layer.k) == 0;
								if (ascending) {
									if (src[i] > src[ixj])
										std::swap(src[i], src[ixj]);
								} else {
									if (src[i] < src[ixj])
										std::swap(src[i], src[ixj]);
								}
							}
						}
					}
				} else {
					// LayerType::Truncate
					if (!try_run_simd_layer_truncate(src, dst, begin, end, layer.j, active_n)) {
						std::size_t i = begin;

						const std::size_t j_minus_1 = layer.j - 1;
						const std::size_t j_mask = ~j_minus_1;

						while (i < end) {
							if ((i & layer.j) != 0) {
								i = (i | ((layer.j << 1) - 1)) + 1;
								continue;
							}

							std::size_t chunk_end = std::min((i | j_minus_1) + 1, end);
							if (chunk_end > active_n)
								chunk_end = active_n;

							for (; i < chunk_end; ++i) {
								const std::size_t ixj = i + layer.j;
								// Fast bitwise truncation
								const std::size_t out_idx = ((i >> 1) & j_mask) | (i & j_minus_1);
								dst[out_idx] = std::min(src[i], src[ixj]);
							}
						}
					}
				}

				barrier.wait();

				// Swap active buffers! Thread 0 does this safely for the group.
				if (tid == 0 && layer.type == common::bitonic::LayerType::Truncate) {
					std::swap(src, dst);
				}

				barrier.wait();
			}
		});
	}

	for (auto& t : pool) {
		t.join();
	}

	// If there was an odd number of truncate operations, the final result is in alt_buffer.
	// We copy it back to original data vector as that's expected by the top-level runner.
	if (src != data.data()) {
		std::copy(alt_buffer.begin(), alt_buffer.end(), data.begin());
	}
}

template void run_topk<std::int32_t>(std::vector<std::int32_t>& data, const std::vector<common::bitonic::Layer>& layers,
									 std::size_t workers);
template void run_topk<std::uint32_t>(std::vector<std::uint32_t>& data,
									  const std::vector<common::bitonic::Layer>& layers, std::size_t workers);
template void run_topk<float>(std::vector<float>& data, const std::vector<common::bitonic::Layer>& layers,
							  std::size_t workers);
template void run_topk<double>(std::vector<double>& data, const std::vector<common::bitonic::Layer>& layers,
							   std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template void run_topk<_Float16>(std::vector<_Float16>& data, const std::vector<common::bitonic::Layer>& layers,
								 std::size_t workers);
#endif

} // namespace cpu::bitonic
