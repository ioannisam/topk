#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common/config.hpp"
#include "common/stats.hpp"

namespace common::reporting {

const char* dtype_name(common::config::DataType dtype);
const char* algorithm_name(common::config::Algorithm algorithm);
std::string format_fixed(double value, int decimals, const char* suffix = "");

void print_section_header(const char* title);
void print_key_value(const char* key, const std::string& value);
void print_key_value(const char* key, const char* value);
void print_key_value(const char* key, std::size_t value);
void print_key_value(const char* key, double value, int precision = 3);

void print_configuration(const common::config::Config& cfg, std::size_t n, std::optional<std::size_t> ex_threads,
						 const char* run_mode_label, const char* backend_tag = nullptr);

void print_timing_lines(const std::vector<std::pair<std::string, std::optional<double>>>& lines);

void print_energy_lines(const common::energy::Summary& energy);

void print_traffic_lines(const common::topk::TrafficStats& traffic);

void print_bitonic_common_metrics(const common::config::Config& cfg, const common::topk::BitonicRunStats& stats);

void print_check_result(const char* label, bool enabled, bool ok);

void print_output(const common::config::Config& cfg, const std::vector<std::string>& output);

} // namespace common::reporting
