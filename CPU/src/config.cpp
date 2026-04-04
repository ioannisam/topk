#include "config.hpp"

#include <stdexcept>
#include <string>

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef CHECK
#define CHECK 1
#endif

namespace {

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

} // namespace

Config parse_args(int argc, char** argv) {

	if (argc < 2 || argc > 10) {
		throw std::invalid_argument("Usage: ./topk <q> [k] [seed] [min|max] [sort|nosort] "
									"[debug|nodebug] [check|nocheck] [threads=<num>]");
	}

	const int q = std::stoi(argv[1]);
	if (q < 0) {
		throw std::invalid_argument("q must be non-negative");
	}

	const std::size_t n = std::size_t{1} << q;
	std::size_t k = n;
	std::uint64_t seed = 42;
	bool want_max = true;
	bool sort_output = false;
	bool debug_output = DEBUG != 0;
	bool run_check = CHECK != 0;
	std::size_t ex_threads = 0;
	int numeric_seen = 0;

	for (int i = 2; i < argc; i++) {
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
			run_check = (token == "check");
			continue;
		}
		if (starts_with(token, "threads=")) {
			const std::string value = token.substr(std::string("threads=").size());
			const long long parsed_threads = std::stoll(value);
			if (parsed_threads <= 0) {
				throw std::invalid_argument("threads must be positive");
			}
			ex_threads = static_cast<std::size_t>(parsed_threads);
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

	return Config{q, k, seed, want_max, sort_output, debug_output, run_check, ex_threads};
}
