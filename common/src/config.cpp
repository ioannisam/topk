#include "common/config.hpp"

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DEBUG
#define DEBUG 0
#endif

namespace common::config {
namespace {

DataType parse_dtype(const std::string& token);
Algorithm parse_algorithm(const std::string& token);
RunMode parse_run_mode(const std::string& token);

bool starts_with(const std::string& text, const std::string& prefix) {
	return text.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& text, const std::string& suffix) {
	if (suffix.size() > text.size()) {
		return false;
	}
	return text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_dtype_token(const std::string& token) {
	return starts_with(token, "dtype=");
}

bool is_key_value_token(const std::string& token) {
	return token.find('=') != std::string::npos;
}

long long parse_signed_long(const std::string& text, const char* field_name) {
	try {
		std::size_t pos = 0;
		const long long value = std::stoll(text, &pos);
		if (pos != text.size()) {
			throw std::invalid_argument("");
		}
		return value;
	} catch (const std::exception&) {
		throw std::invalid_argument(std::string(field_name) + " must be an integer");
	}
}

bool parse_bool_value(const std::string& value, const char* field_name) {
	if (value == "true" || value == "1" || value == "yes" || value == "on") {
		return true;
	}
	if (value == "false" || value == "0" || value == "no" || value == "off") {
		return false;
	}
	throw std::invalid_argument(std::string(field_name) + " must be true/false");
}

bool is_integer_token(const std::string& token) {
	if (token.empty()) {
		return false;
	}

	std::size_t pos = 0;
	if (token[0] == '+' || token[0] == '-') {
		if (token.size() == 1) {
			return false;
		}
		pos = 1;
	}

	for (; pos < token.size(); ++pos) {
		if (!std::isdigit(static_cast<unsigned char>(token[pos]))) {
			return false;
		}
	}

	return true;
}

std::string to_lower(std::string text) {
	for (char& c : text) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	return text;
}

std::string trim(const std::string& text) {
	std::size_t begin = 0;
	while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
		++begin;
	}

	if (begin == text.size()) {
		return "";
	}

	std::size_t end = text.size();
	while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
		--end;
	}

	return text.substr(begin, end - begin);
}

std::vector<std::string> split_ws(const std::string& text) {
	std::vector<std::string> out;
	std::string current;

	for (char c : text) {
		if (std::isspace(static_cast<unsigned char>(c))) {
			if (!current.empty()) {
				out.push_back(current);
				current.clear();
			}
		} else {
			current.push_back(c);
		}
	}

	if (!current.empty()) {
		out.push_back(current);
	}

	return out;
}

Config parse_tokens(const std::vector<std::string>& tokens) {

	if (tokens.empty() || tokens.size() > 15) {
		throw std::invalid_argument("Usage: ./topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] "
									"[algo=bitonic|map_reduce] [run=full|trunc|both] [debug=true|false] "
									"[threads=<num>] [seed=<seed>] "
									"[verify=true|false] [min=<int>] [max=<int>] (key=value only)");
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
			rand_min = static_cast<int>(parse_signed_long(token.substr(4), "min"));
			continue;
		}
		if (starts_with(token, "max=")) {
			rand_max = static_cast<int>(parse_signed_long(token.substr(4), "max"));
			continue;
		}
		if (starts_with(token, "run=")) {
			run_mode = parse_run_mode(token.substr(4));
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
	if (value == "fp16") {
		return DataType::Fp16;
	}
	throw std::invalid_argument("Unsupported dtype. Use one of: int, uint, float, double, fp16");
}

Algorithm parse_algorithm(const std::string& token) {
	const std::string value = token.substr(std::string("algo=").size());
	if (value == "bitonic") {
		return Algorithm::Bitonic;
	}
	if (value == "map_reduce" || value == "mapreduce") {
		return Algorithm::MapReduce;
	}
	throw std::invalid_argument("Unsupported algorithm. Use one of: bitonic, map_reduce");
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

} // namespace

Config parse_args(int argc, char** argv) {
	if (argc < 2) {
		throw std::invalid_argument("Usage: ./topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] "
									"[algo=bitonic|map_reduce] [run=full|trunc|both] [debug=true|false] "
									"[threads=<num>] [seed=<seed>] [verify=true|false] [min=<int>] [max=<int>]");
	}

	if (argc > 16) {
		throw std::invalid_argument("Too many arguments. Expected at most 15 CLI tokens after program name.");
	}

	std::vector<std::string> cli_tokens;
	cli_tokens.reserve(static_cast<std::size_t>(argc - 1));
	for (int i = 1; i < argc; ++i) {
		cli_tokens.push_back(argv[i]);
	}

	return parse_tokens(cli_tokens);
}

} // namespace common::config
