#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "common/config.hpp"

namespace common::utils {

template <typename T> std::string format_value(T value) {
	if constexpr (std::is_integral_v<T>) {
		if constexpr (std::is_unsigned_v<T>) {
			return std::to_string(static_cast<unsigned long long>(value));
		}
		return std::to_string(static_cast<long long>(value));
	}

	std::ostringstream out;
	out << static_cast<double>(value);
	return out.str();
}

template <typename T> std::vector<std::string> format_output(const std::vector<T>& values) {
	std::vector<std::string> out;
	out.reserve(values.size());
	for (const T value : values) {
		out.push_back(format_value(value));
	}
	return out;
}

template <typename T> T parse_expected_value(const std::string& token) {
	if constexpr (std::is_integral_v<T>) {
		if constexpr (std::is_unsigned_v<T>) {
			return static_cast<T>(std::stoull(token));
		}
		return static_cast<T>(std::stoll(token));
	}

	return static_cast<T>(std::stod(token));
}

template <typename T> bool value_equal(T lhs, T rhs) {
	if constexpr (std::is_floating_point_v<T>) {
		const double a = static_cast<double>(lhs);
		const double b = static_cast<double>(rhs);
		const double diff = std::fabs(a - b);
		const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
		const double rel_tol = 1e-6 * scale;
		const double abs_tol = 1e-3;
		return diff <= std::max(rel_tol, abs_tol);
	}
	return lhs == rhs;
}

template <typename T> bool validate_expected_output(const config::Config& cfg, const std::vector<T>& output) {
	if (!cfg.has_expected_output) {
		return true;
	}

	if (cfg.expected_output_tokens.size() != output.size()) {
		return false;
	}

	for (std::size_t i = 0; i < output.size(); ++i) {
		const T expected = parse_expected_value<T>(cfg.expected_output_tokens[i]);
		if (!value_equal(output[i], expected)) {
			return false;
		}
	}

	return true;
}

} // namespace common::utils