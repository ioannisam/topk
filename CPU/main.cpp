#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

constexpr int randomMin = 0;
constexpr int randomMax = 1000;

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef CHECK
#define CHECK 1
#endif

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

struct Layer {
	std::size_t k;
	std::size_t j;
};

struct Config {
	int q;
	int p;
	std::size_t k;
	std::uint64_t seed;
	bool want_max;
	bool sort_output;
	bool debug_output;
	bool run_full_check;
	std::size_t exec_threads;
};

bool is_mode_token(const std::string& token) {
	return token == "min" || token == "max";
}

bool is_sort_token(const std::string& token) {
	return token == "sort" || token == "nosort";
}

bool is_debug_token(const std::string& token) {
	return token == "debug" || token == "nodebug";
}

bool is_check_token(const std::string& token) {
	return token == "check" || token == "nocheck";
}

bool starts_with(const std::string& text, const std::string& prefix) {
	return text.rfind(prefix, 0) == 0;
}

Config parse_args(int argc, char** argv) {

	if (argc < 3 || argc > 11) {
		throw std::invalid_argument("Usage: ./topk <q> <p> [k] [seed] [min|max] [sort|nosort] "
									"[debug|nodebug] [check|nocheck] [threads=<num>]");
	}

	const int q = std::stoi(argv[1]);
	const int p = std::stoi(argv[2]);
	if (q < 0 || p < 0) {
		throw std::invalid_argument("q and p must be non-negative");
	}

	const std::size_t n = std::size_t{1} << (q + p);
	std::size_t k = n;
	std::uint64_t seed = 42;
	bool want_max = true;
	bool sort_output = false;
	bool debug_output = DEBUG != 0;
	bool run_full_check = CHECK != 0;
	std::size_t exec_threads = 0;
	int numeric_seen = 0;

	for (int i = 3; i < argc; i++) {
		const std::string token = argv[i];
		if (is_mode_token(token)) {
			want_max = (token == "max");
			continue;
		}
		if (is_sort_token(token)) {
			sort_output = (token == "sort");
			continue;
		}
		if (is_debug_token(token)) {
			debug_output = (token == "debug");
			continue;
		}
		if (is_check_token(token)) {
			run_full_check = (token == "check");
			continue;
		}
		if (starts_with(token, "threads=")) {
			const std::string value = token.substr(std::string("threads=").size());
			const long long parsed_threads = std::stoll(value);
			if (parsed_threads <= 0) {
				throw std::invalid_argument("threads must be positive");
			}
			exec_threads = static_cast<std::size_t>(parsed_threads);
			continue;
		}

		const long long parsed = std::stoll(token);
		if (numeric_seen == 0) {
			if (parsed <= 0) {
				throw std::invalid_argument("k must be positive");
			}
			k = static_cast<std::size_t>(parsed);
			if (k > n) {
				k = n;
			}
		} else if (numeric_seen == 1) {
			if (parsed < 0) {
				throw std::invalid_argument("seed must be non-negative");
			}
			seed = static_cast<std::uint64_t>(parsed);
		} else {
			throw std::invalid_argument("Too many numeric arguments. Expected at most: [k] [seed]");
		}
		numeric_seen++;
	}

	return Config{q, p, k, seed, want_max, sort_output, debug_output, run_full_check, exec_threads};
}

std::vector<Layer> build_layers(std::size_t n) {
	std::vector<Layer> layers;
	for (std::size_t k = 2; k <= n; k <<= 1) {
		for (std::size_t j = k >> 1; j > 0; j >>= 1) {
			layers.push_back(Layer{k, j});
		}
	}
	return layers;
}

std::vector<std::vector<unsigned char>> build_keep_masks(const std::vector<Layer>& layers, std::size_t n,
														 std::size_t k) {

	std::vector<std::vector<unsigned char>> keep(layers.size(), std::vector<unsigned char>(n, 0));
	std::vector<unsigned char> needed(n, 0);
	for (std::size_t i = 0; i < k; ++i) {
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

void run_network_parallel(std::vector<int>& data, const std::vector<Layer>& layers,
						  const std::vector<std::vector<unsigned char>>& keep, bool truncated, std::size_t workers) {

	const std::size_t n = data.size();
	Barrier barrier(workers);
	std::vector<std::thread> pool;
	pool.reserve(workers);

	for (std::size_t tid = 0; tid < workers; ++tid) {
		pool.emplace_back([&, tid]() {
			const std::size_t begin = (n * tid) / workers;
			const std::size_t end = (n * (tid + 1)) / workers;

			for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
				const std::size_t k = layers[layer_idx].k;
				const std::size_t j = layers[layer_idx].j;

				for (std::size_t i = begin; i < end; ++i) {
					const std::size_t ixj = i ^ j;
					if (ixj <= i || ixj >= n) {
						continue;
					}

					if (truncated && !(keep[layer_idx][i] || keep[layer_idx][ixj])) {
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

std::size_t count_full_comparators(const std::vector<Layer>& layers, std::size_t n) {
	return layers.size() * (n / 2);
}

std::size_t count_truncated_comparators(const std::vector<Layer>& layers,
										const std::vector<std::vector<unsigned char>>& keep, std::size_t n) {

	std::size_t active = 0;
	for (std::size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
		const std::size_t j = layers[layer_idx].j;
		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t ixj = i ^ j;
			if (ixj <= i || ixj >= n) {
				continue;
			}
			if (keep[layer_idx][i] || keep[layer_idx][ixj]) {
				active++;
			}
		}
	}
	return active;
}

} // namespace

int main(int argc, char** argv) {

	try {
		const Config cfg = parse_args(argc, argv);

		const std::size_t virtual_workers = std::size_t{1} << cfg.p;
		const std::size_t local = std::size_t{1} << cfg.q;
		const std::size_t n = virtual_workers * local;
		const std::size_t hw_threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
		std::size_t exec_threads = cfg.exec_threads == 0 ? hw_threads : cfg.exec_threads;
		exec_threads = std::max<std::size_t>(1, std::min(exec_threads, n));

		std::cout << "Virtual ranks (2^p): " << virtual_workers << ", execution threads: " << exec_threads
				  << ", local chunk: " << local << ", N: " << n << "\n";
		std::cout << "Requested top-k: " << cfg.k << "\n";
		std::cout << "Mode: " << (cfg.want_max ? "max" : "min") << "\n";
		std::cout << "Final sort: " << (cfg.sort_output ? "on" : "off") << "\n";
		std::cout << "Debug output: " << (cfg.debug_output ? "on" : "off") << "\n";
		std::cout << "Full reference check: " << (cfg.run_full_check ? "on" : "off") << "\n";
		auto layers = build_layers(n);
		auto keep = build_keep_masks(layers, n, cfg.k);

		const std::size_t full_cmp = count_full_comparators(layers, n);
		const std::size_t trunc_cmp = count_truncated_comparators(layers, keep, n);

		std::mt19937 rng(static_cast<std::mt19937::result_type>(cfg.seed));
		std::uniform_int_distribution<int> dist(randomMin, randomMax);
		std::vector<int> input(n);
		for (std::size_t i = 0; i < n; i++) {
			input[i] = dist(rng);
		}

		if (cfg.want_max) {
			for (std::size_t i = 0; i < n; i++) {
				input[i] = -input[i];
			}
		}

		std::vector<int> truncated = input;
		auto t2 = std::chrono::high_resolution_clock::now();
		run_network_parallel(truncated, layers, keep, true, exec_threads);
		auto t3 = std::chrono::high_resolution_clock::now();

		bool ok = true;
		double full_ms = 0.0;
		if (cfg.run_full_check) {
			std::vector<int> full = input;
			auto t0 = std::chrono::high_resolution_clock::now();
			run_network_parallel(full, layers, keep, false, exec_threads);
			auto t1 = std::chrono::high_resolution_clock::now();
			full_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

			for (std::size_t i = 0; i < cfg.k; i++) {
				if (truncated[i] != full[i]) {
					ok = false;
					break;
				}
			}
		}
		const double trunc_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

		if (cfg.run_full_check) {
			std::cout << "Full bitonic time (ms): " << full_ms << "\n";
		} else {
			std::cout << "Full bitonic time (ms): skipped\n";
		}
		std::cout << "Truncated bitonic time (ms): " << trunc_ms << "\n";

		if (cfg.debug_output) {
			const double skipped_pct =
				full_cmp == 0 ? 0.0
							  : (100.0 * static_cast<double>(full_cmp - trunc_cmp) / static_cast<double>(full_cmp));
			const double speedup = (cfg.run_full_check && trunc_ms > 0.0) ? (full_ms / trunc_ms) : 0.0;
			std::cout << "[debug] Hardware threads available: " << hw_threads << "\n";
			std::cout << "[debug] Execution threads used: " << exec_threads << "\n";
			std::cout << "[debug] Virtual bitonic ranks: " << virtual_workers << "\n";
			std::cout << "[debug] Bitonic layers: " << layers.size() << "\n";
			std::cout << "[debug] Full comparators: " << full_cmp << "\n";
			std::cout << "[debug] Truncated comparators: " << trunc_cmp << "\n";
			std::cout << "[debug] Comparator skip ratio (%): " << skipped_pct << "\n";
			if (cfg.run_full_check) {
				std::cout << "[debug] Full/Truncated speedup: " << speedup << "x\n";
			}
		}

		if (cfg.run_full_check) {
			std::cout << "Top-k correctness vs full network: " << (ok ? "OK" : "FAIL") << "\n";
			if (!ok) {
				return 2;
			}
		}

		std::vector<int> output(cfg.k);
		for (std::size_t i = 0; i < cfg.k; i++) {
			output[i] = cfg.want_max ? -truncated[i] : truncated[i];
		}

		if (cfg.sort_output) {
			if (cfg.want_max) {
				std::sort(output.begin(), output.end(), std::greater<int>());
			} else {
				std::sort(output.begin(), output.end());
			}
		}

		std::cout << "Top-k output (" << (cfg.want_max ? "max" : "min") << ", "
				  << (cfg.sort_output ? "sorted" : "network-order") << "):";
		for (int value : output) {
			std::cout << ' ' << value;
		}
		std::cout << "\n";

		return 0;
	} catch (const std::exception& ex) {

		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}
}
