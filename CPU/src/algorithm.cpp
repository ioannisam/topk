#include "algorithm.hpp"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace {

class Barrier {

  public:
	explicit Barrier(std::size_t participants) : threshold(participants), count(participants), generation(0) {
	}

	void wait() {
		std::unique_lock<std::mutex> lock(mutex);
		const std::size_t gen = generation;
		if (--count == 0) {
			generation++;
			count = threshold;
			cv.notify_all();
			return;
		}
		cv.wait(lock, [&] { return generation != gen; });
	}

  private:
	std::mutex mutex;
	std::condition_variable cv;
	std::size_t threshold;
	std::size_t count;
	std::size_t generation;
};

} // namespace

std::vector<std::vector<unsigned char>> build_masks(const std::vector<Layer>& layers, std::size_t n, std::size_t k) {

	std::vector<std::vector<unsigned char>> keep(layers.size(), std::vector<unsigned char>(n, 0));
	std::vector<unsigned char> needed(n, 0);
	for (std::size_t i = 0; i < k; i++) {
		needed[i] = 1;
	}

	for (std::size_t idx = layers.size(); idx-- > 0;) {
		const std::size_t j = layers[idx].j;
		std::vector<unsigned char> prev_needed = needed;

		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t ixj = i ^ j;
			if (ixj <= i) {
				continue;
			}
			if (needed[i] || needed[ixj]) {
				keep[idx][i] = 1;
				keep[idx][ixj] = 1;
				prev_needed[i] = 1;
				prev_needed[ixj] = 1;
			}
		}

		needed.swap(prev_needed);
	}

	return keep;
}

template <typename T>
void run_network_parallel(std::vector<T>& data, const std::vector<Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool trunc, std::size_t workers) {

	const std::size_t n = data.size();
	Barrier barrier(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers);

	for (std::size_t tid = 0; tid < workers; tid++) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;

			for (std::size_t layer_idx = 0; layer_idx < layers.size(); layer_idx++) {
				const std::size_t k = layers[layer_idx].k;
				const std::size_t j = layers[layer_idx].j;

				for (std::size_t i = begin; i < end; ++i) {
					const std::size_t ixj = i ^ j;
					if (ixj <= i || ixj >= n) {
						continue;
					}

					if (trunc && !(keep[layer_idx][i] || keep[layer_idx][ixj])) {
						continue;
					}

					const bool ascending = (i & k) == 0;
					if (ascending) {
						if (data[i] > data[ixj]) {
							std::swap(data[i], data[ixj]);
						}
					} else {
						if (data[i] < data[ixj]) {
							std::swap(data[i], data[ixj]);
						}
					}
				}

				barrier.wait();
			}
		});
	}

	for (auto& t : pool) {
		t.join();
	}
}

template void run_network_parallel<std::int32_t>(std::vector<std::int32_t>& data, const std::vector<Layer>& layers,
												 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
												 std::size_t workers);
template void run_network_parallel<std::uint32_t>(std::vector<std::uint32_t>& data, const std::vector<Layer>& layers,
												  const std::vector<std::vector<unsigned char>>& keep, bool trunc,
												  std::size_t workers);
template void run_network_parallel<float>(std::vector<float>& data, const std::vector<Layer>& layers,
										  const std::vector<std::vector<unsigned char>>& keep, bool trunc,
										  std::size_t workers);
template void run_network_parallel<double>(std::vector<double>& data, const std::vector<Layer>& layers,
										   const std::vector<std::vector<unsigned char>>& keep, bool trunc,
										   std::size_t workers);

#if defined(__FLT16_MANT_DIG__)
template void run_network_parallel<_Float16>(std::vector<_Float16>& data, const std::vector<Layer>& layers,
											 const std::vector<std::vector<unsigned char>>& keep, bool trunc,
											 std::size_t workers);
#endif
