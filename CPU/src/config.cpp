#include "config.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DEBUG
#define DEBUG 0
#endif

namespace {

struct ParsedCase {
	std::vector<std::string> args;
	std::vector<std::string> expected;
	bool has_expected;
};

DataType parse_dtype(const std::string& token);
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

std::vector<std::string> strip_program_name(std::vector<std::string> args) {
	if (args.empty()) {
		return args;
	}

	const std::string& first = args.front();
	if (first == "topk" || ends_with(first, "/topk") || ends_with(first, "\\topk") || ends_with(first, "/topk.exe") ||
		ends_with(first, "\\topk.exe")) {
		args.erase(args.begin());
	}

	return args;
}

ParsedCase parse_testcase(const std::string& testcase_path) {
	std::ifstream in(testcase_path);
	if (!in.is_open()) {
		throw std::invalid_argument("Could not open testcase file: " + testcase_path);
	}

	std::vector<std::string> free_lines;
	std::vector<std::string> command_args;
	std::vector<std::string> expected_tokens;

	std::string raw;
	while (std::getline(in, raw)) {
		const std::string line = trim(raw);
		if (line.empty() || line[0] == '#') {
			continue;
		}

		const std::string lower = to_lower(line);
		if (starts_with(lower, "command:")) {
			command_args = split_ws(trim(line.substr(std::string("command:").size())));
			continue;
		}
		if (starts_with(lower, "args:")) {
			command_args = split_ws(trim(line.substr(std::string("args:").size())));
			continue;
		}
		if (starts_with(lower, "answer:")) {
			expected_tokens = split_ws(trim(line.substr(std::string("answer:").size())));
			continue;
		}
		if (starts_with(lower, "expected:")) {
			expected_tokens = split_ws(trim(line.substr(std::string("expected:").size())));
			continue;
		}

		free_lines.push_back(line);
	}

	if (command_args.empty() && !free_lines.empty()) {
		command_args = split_ws(free_lines.front());
	}
	if (expected_tokens.empty() && free_lines.size() >= 2) {
		expected_tokens = split_ws(free_lines.back());
	}

	command_args = strip_program_name(command_args);
	if (command_args.empty()) {
		throw std::invalid_argument("Testcase file must include a command line with at least q as the first argument");
	}

	return ParsedCase{command_args, expected_tokens, !expected_tokens.empty()};
}

Config parse_tokens(const std::vector<std::string>& tokens, const std::string& testcase_path,
					const std::vector<std::string>& expected_output_tokens, bool has_expected_output,
					bool from_testcase) {

	if (tokens.empty() || tokens.size() > 11) {
		throw std::invalid_argument("Usage: ./topk q=<q> [k=<k>] [mode=min|max] [dtype=<type>] "
									"[run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>] "
									"[check=true|false] (testcase mode only, key=value only)");
	}

	int q = -1;
	bool q_seen = false;
	long long k_raw = -1;
	bool k_seen = false;
	std::uint64_t seed = 42;
	bool want_max = true;
	bool debug_output = DEBUG != 0;
	bool run_check = false;
	RunMode run_mode = RunMode::Trunc;
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
		if (starts_with(token, "debug=")) {
			debug_output = parse_bool_value(token.substr(6), "debug");
			continue;
		}
		if (starts_with(token, "check=")) {
			if (!from_testcase) {
				throw std::invalid_argument("check=true/false is only supported when running from a testcase file");
			}
			run_check = parse_bool_value(token.substr(6), "check");
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

	const std::size_t n = std::size_t{1} << q;
	std::size_t k = n;
	if (k_seen) {
		k = static_cast<std::size_t>(k_raw);
		if (k > n) {
			k = n;
		}
	}

	if (run_check && !has_expected_output) {
		throw std::invalid_argument("check requires an expected answer line in the testcase file");
	}

	Config cfg{};
	cfg.q = q;
	cfg.k = k;
	cfg.want_max = want_max;
	cfg.dtype = dtype;
	cfg.run_mode = run_mode;
	cfg.debug_output = debug_output;
	cfg.ex_threads = ex_threads;
	cfg.seed = seed;
	cfg.testcase_path = testcase_path;
	cfg.run_check = run_check;
	cfg.expected_output_tokens = expected_output_tokens;
	cfg.has_expected_output = has_expected_output;
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
									"[run=full|trunc|both] [debug=true|false] [threads=<num>] [seed=<seed>]\n"
									"   or: ./topk <testcase-file>\n"
									"   or: ./topk --case <testcase-file>\n"
									"   or: ./topk case=<testcase-file>\n"
									"In testcase mode only: [check=true|false]");
	}

	std::vector<std::string> cli_tokens;
	cli_tokens.reserve(static_cast<std::size_t>(argc - 1));
	for (int i = 1; i < argc; ++i) {
		cli_tokens.push_back(argv[i]);
	}

	std::string testcase_path;
	if (cli_tokens.size() == 2 && cli_tokens[0] == "--case") {
		testcase_path = cli_tokens[1];
	} else if (cli_tokens.size() == 1 && starts_with(cli_tokens[0], "case=")) {
		testcase_path = cli_tokens[0].substr(std::string("case=").size());
	} else if (cli_tokens.size() == 1 && starts_with(cli_tokens[0], "--case=")) {
		testcase_path = cli_tokens[0].substr(std::string("--case=").size());
	} else if (cli_tokens.size() == 1 && !is_integer_token(cli_tokens[0]) && std::filesystem::exists(cli_tokens[0])) {
		testcase_path = cli_tokens[0];
	}

	if (!testcase_path.empty()) {
		const ParsedCase testcase = parse_testcase(testcase_path);
		return parse_tokens(testcase.args, testcase_path, testcase.expected, testcase.has_expected, true);
	}

	if (argc > 12) {
		throw std::invalid_argument("Too many arguments. Expected at most 11 CLI tokens after program name.");
	}

	return parse_tokens(cli_tokens, "", {}, false, false);
}
