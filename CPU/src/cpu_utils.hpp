#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace cpu::utils {

class WorkerPool {
  public:
	explicit WorkerPool(std::size_t workers) : num_workers(workers) {
		threads.reserve(workers);
		for (std::size_t tid = 0; tid < workers; tid++) {
			threads.emplace_back([this, tid]() { loop(tid); });
		}
	}

	~WorkerPool() {
		alive.store(false, std::memory_order_relaxed);
		gen.fetch_add(1, std::memory_order_release);
		for (auto& t : threads) {
			t.join();
		}
	}

	WorkerPool(const WorkerPool&) = delete;
	WorkerPool& operator=(const WorkerPool&) = delete;

	std::size_t size() const {
		return num_workers;
	}

	void run(std::function<void(std::size_t)> fn) {
		task = std::move(fn);
		done.store(0, std::memory_order_relaxed);
		gen.fetch_add(1, std::memory_order_release);
		while (done.load(std::memory_order_acquire) < num_workers) {
#if defined(__x86_64__) || defined(__i386__)
			_mm_pause();
#else
			std::this_thread::yield();
#endif
		}
	}

  private:
	void loop(std::size_t tid) {
		std::size_t seen = 0;
		while (true) {
			std::size_t g;
			while ((g = gen.load(std::memory_order_acquire)) == seen) {
#if defined(__x86_64__) || defined(__i386__)
				_mm_pause();
#else
				std::this_thread::yield();
#endif
			}
			seen = g;
			if (!alive.load(std::memory_order_relaxed)) {
				return;
			}
			task(tid);
			done.fetch_add(1, std::memory_order_acq_rel);
		}
	}

	std::size_t num_workers;
	std::vector<std::thread> threads;
	std::function<void(std::size_t)> task;
	alignas(64) std::atomic<std::size_t> gen{0};
	alignas(64) std::atomic<std::size_t> done{0};
	std::atomic<bool> alive{true};
};

} // namespace cpu::utils
