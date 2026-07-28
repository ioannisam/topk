#pragma once

#include <limits>
#include <stdexcept>
#include <string>

namespace common::parse {

inline bool starts_with(const std::string& text, const std::string& prefix) {
	return text.rfind(prefix, 0) == 0;
}

inline long long parse_signed_long(const std::string& text, const char* field_name) {
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

inline int parse_int_field(const std::string& text, const char* field_name) {
	const long long value = parse_signed_long(text, field_name);
	if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
		throw std::invalid_argument(std::string(field_name) + " must fit in a 32-bit int");
	}
	return static_cast<int>(value);
}

inline bool parse_bool_value(const std::string& value, const char* field_name) {
	if (value == "true" || value == "1" || value == "yes" || value == "on") {
		return true;
	}
	if (value == "false" || value == "0" || value == "no" || value == "off") {
		return false;
	}
	throw std::invalid_argument(std::string(field_name) + " must be true/false");
}

} // namespace common::parse
