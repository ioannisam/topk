#include "common/config.hpp"

#include "parse_utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DEBUG
#define DEBUG 0
#endif

namespace common::config {
namespace {

// n = 1 << q must stay a defined shift on std::size_t.
constexpr long long kMaxQ = 63;

// Largest finite magnitude representable by IEEE binary16.
constexpr int kHalfMax = 65504;

constexpr const char* kUsage = "Usage: ./topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] "
							   "[algo=bitonic|map_reduce|gt] [run=full|trunc|both] [debug=true|false] "
							   "[threads=<num>] [seed=<seed>] [verify=true|false] [min=<int>] [max=<int>] "
							   "[dist=uniform|normal|trimodal|sorted|reverse|adversarial]";

DataType parse_dtype(const std::string& token);
Algorithm parse_algorithm(const std::string& token);
RunMode parse_run_mode(const std::string& token);
Distribution parse_distribution(const std::string& token);

using common::parse::parse_bool_value;
using common::parse::parse_int_field;
using common::parse::parse_signed_long;
using common::parse::starts_with;

bool is_dtype_token(const std::string& token) {
	return starts_with(token, "dtype=");
}

bool is_key_value_token(const std::string& token) {
	return token.find('=') != std::string::npos;
}

Config parse_tokens(const std::vector<std::string>& tokens) {

	if (tokens.empty() || tokens.size() > 15) {
		throw std::invalid_argument(kUsage);
	}

	int q = -1;
	bool q_seen = false;
	long long k_raw = -1;
	bool k_seen = false;
	std::uint64_t seed = 42;
	bool want_max = true;
	bool debug_output = DEBUG != 0;
	bool verify_output = false;
	int rand_min = 0;
	int rand_max = 1000;
	RunMode run_mode = RunMode::Trunc;
	Algorithm algorithm = Algorithm::Bitonic;
	std::size_t ex_threads = 0;
	DataType dtype = DataType::Int;
	Distribution dist = Distribution::Uniform;

	for (std::size_t i = 0; i < tokens.size(); i++) {
		const std::string& token = tokens[i];
		if (!is_key_value_token(token)) {
			throw std::invalid_argument("All arguments must use key=value format. Unknown token: " + token);
		}

		if (starts_with(token, "q=")) {
			const long long parsed_q = parse_signed_long(token.substr(2), "q");
			if (parsed_q < 0) {
				throw std::invalid_argument("q must be non-negative");
			}
			if (parsed_q > kMaxQ) {
				throw std::invalid_argument("q must be <= " + std::to_string(kMaxQ));
			}
			q = static_cast<int>(parsed_q);
			q_seen = true;
			continue;
		}
		if (starts_with(token, "k=")) {
			const long long parsed_k = parse_signed_long(token.substr(2), "k");
			if (parsed_k <= 0) {
				throw std::invalid_argument("k must be positive");
			}
			k_raw = parsed_k;
			k_seen = true;
			continue;
		}
		if (starts_with(token, "seed=")) {
			const long long parsed_seed = parse_signed_long(token.substr(5), "seed");
			if (parsed_seed < 0) {
				throw std::invalid_argument("seed must be non-negative");
			}
			seed = static_cast<std::uint64_t>(parsed_seed);
			continue;
		}
		if (starts_with(token, "mode=")) {
			const std::string value = token.substr(5);
			if (value == "min" || value == "max") {
				want_max = (value == "max");
				continue;
			}
			throw std::invalid_argument("mode must be min or max");
		}
		if (starts_with(token, "threads=")) {
			const long long parsed_threads = parse_signed_long(token.substr(8), "threads");
			if (parsed_threads <= 0) {
				throw std::invalid_argument("threads must be positive");
			}
			ex_threads = static_cast<std::size_t>(parsed_threads);
			continue;
		}
		if (is_dtype_token(token)) {
			dtype = parse_dtype(token);
			continue;
		}
		if (starts_with(token, "algo=")) {
			algorithm = parse_algorithm(token);
			continue;
		}
		if (starts_with(token, "debug=")) {
			debug_output = parse_bool_value(token.substr(6), "debug");
			continue;
		}
		if (starts_with(token, "verify=")) {
			verify_output = parse_bool_value(token.substr(7), "verify");
			continue;
		}
		if (starts_with(token, "min=")) {
			rand_min = parse_int_field(token.substr(4), "min");
			continue;
		}
		if (starts_with(token, "max=")) {
			rand_max = parse_int_field(token.substr(4), "max");
			continue;
		}
		if (starts_with(token, "run=")) {
			run_mode = parse_run_mode(token.substr(4));
			continue;
		}
		if (starts_with(token, "dist=")) {
			dist = parse_distribution(token.substr(5));
			continue;
		}

		throw std::invalid_argument("Unknown key token: " + token);
	}

	if (!q_seen) {
		throw std::invalid_argument("q is required. Use q=<non-negative-int>");
	}

	if (rand_max < rand_min) {
		throw std::invalid_argument("max must be >= min");
	}

	// Clamp here rather than in the generator so the reported range is the one actually sampled.
	if (dtype == DataType::Half) {
		rand_min = std::clamp(rand_min, -kHalfMax, kHalfMax);
		rand_max = std::clamp(rand_max, -kHalfMax, kHalfMax);
	} else if (dtype == DataType::UInt) {
		rand_min = std::max(0, rand_min);
		rand_max = std::max(0, rand_max);
	}

	const std::size_t n = std::size_t{1} << q;
	std::size_t k = n;
	if (k_seen) {
		k = static_cast<std::size_t>(k_raw);
		if (k > n) {
			k = n;
		}
	}

	Config cfg{};
	cfg.q = q;
	cfg.k = k;
	cfg.want_max = want_max;
	cfg.dtype = dtype;
	cfg.algorithm = algorithm;
	cfg.run_mode = run_mode;
	cfg.debug_output = debug_output;
	cfg.ex_threads = ex_threads;
	cfg.seed = seed;
	cfg.verify_output = verify_output;
	cfg.rand_min = rand_min;
	cfg.rand_max = rand_max;
	cfg.dist = dist;
	return cfg;
}

DataType parse_dtype(const std::string& token) {
	const std::string value = token.substr(std::string("dtype=").size());
	if (value == "int") {
		return DataType::Int;
	}
	if (value == "uint") {
		return DataType::UInt;
	}
	if (value == "float") {
		return DataType::Float;
	}
	if (value == "double") {
		return DataType::Double;
	}
	if (value == "half") {
		return DataType::Half;
	}
	throw std::invalid_argument("Unsupported dtype. Use one of: int, uint, float, double, half");
}

Algorithm parse_algorithm(const std::string& token) {
	const std::string value = token.substr(std::string("algo=").size());
	if (value == "bitonic") {
		return Algorithm::Bitonic;
	}
	if (value == "map_reduce" || value == "mapreduce") {
		return Algorithm::MapReduce;
	}
	if (value == "gt") {
		return Algorithm::GroundTruth;
	}
	throw std::invalid_argument("Unsupported algorithm. Use one of: bitonic, map_reduce, gt");
}

RunMode parse_run_mode(const std::string& token) {
	if (token == "trunc") {
		return RunMode::Trunc;
	}
	if (token == "full") {
		return RunMode::Full;
	}
	if (token == "both") {
		return RunMode::Both;
	}
	throw std::invalid_argument("Unsupported run mode. Use one of: full, trunc, both");
}

Distribution parse_distribution(const std::string& token) {
	if (token == "uniform") {
		return Distribution::Uniform;
	}
	if (token == "normal" || token == "gaussian") {
		return Distribution::Normal;
	}
	if (token == "trimodal") {
		return Distribution::Trimodal;
	}
	if (token == "sorted") {
		return Distribution::Sorted;
	}
	if (token == "reverse") {
		return Distribution::Reverse;
	}
	if (token == "adversarial") {
		return Distribution::Adversarial;
	}
	throw std::invalid_argument(
		"Unsupported distribution. Use one of: uniform, normal, trimodal, sorted, reverse, adversarial");
}

} // namespace

Config parse_args(int argc, char** argv) {
	std::vector<std::string> cli_tokens;
	cli_tokens.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0);
	for (int i = 1; i < argc; ++i) {
		cli_tokens.push_back(argv[i]);
	}

	return parse_tokens(cli_tokens);
}

} // namespace common::config
