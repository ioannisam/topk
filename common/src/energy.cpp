#include "common/energy.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

#if defined(TOPK_WITH_NVML)
#include <nvml.h>
#endif

namespace common::energy {

namespace {

bool g_device_requested = false;

double now_seconds() {
	using clock = std::chrono::steady_clock;
	return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

bool read_uint64(const std::string& path, std::uint64_t& out) {
	std::ifstream f(path);
	if (!f) {
		return false;
	}
	std::uint64_t v = 0;
	if (!(f >> v)) {
		return false;
	}
	out = v;
	return true;
}

std::string read_line(const std::string& path) {
	std::ifstream f(path);
	std::string s;
	if (f) {
		std::getline(f, s);
	}
	return s;
}

struct RaplDomain {
	std::string energy_path;
	std::uint64_t range_uj = 0;
	std::uint64_t last_uj = 0;
	double accumulated_j = 0.0;
	bool valid = false;

	bool init(const std::string& dir) {
		energy_path = dir + "/energy_uj";
		std::uint64_t probe = 0;
		if (!read_uint64(energy_path, probe)) {
			return false;
		}
		if (!read_uint64(dir + "/max_energy_range_uj", range_uj) || range_uj == 0) {
			range_uj = ~std::uint64_t{0};
		}
		last_uj = probe;
		valid = true;
		return true;
	}

	double joules() {
		if (!valid) {
			return 0.0;
		}
		std::uint64_t now = 0;
		if (!read_uint64(energy_path, now)) {
			return accumulated_j;
		}
		const std::uint64_t delta = (now >= last_uj) ? (now - last_uj) : (range_uj - last_uj + now);
		last_uj = now;
		accumulated_j += static_cast<double>(delta) * 1e-6;
		return accumulated_j;
	}
};

class SystemCounter final : public Counter {
  public:
	SystemCounter() {
		discover_rapl();
		init_device();
	}

	~SystemCounter() override {
#if defined(TOPK_WITH_NVML)
		sampler_stop.store(true, std::memory_order_relaxed);
		if (sampler.joinable()) {
			sampler.join();
		}
		if (nvml_ready) {
			nvmlShutdown();
		}
#endif
	}

	bool available() const override {
		return package.valid || device_ready;
	}

	Sample read() override {
		Sample s;
		s.package_j = package.joules();
		s.core_j = core.joules();
		s.device_j = device_joules();
		return s;
	}

	std::string describe() const override {
		std::string d;
		d += package.valid ? "rapl:package" : "rapl:unavailable";
		if (core.valid) {
			d += "+core";
		}
		if (device_ready) {
			d += ",nvml:board-integrated";
		}
		return d;
	}

  private:
	void discover_rapl() {
		namespace fs = std::filesystem;
		std::error_code ec;
		const char* root_env = std::getenv("TOPK_RAPL_ROOT");
		const fs::path root(root_env != nullptr ? root_env : "/sys/class/powercap");
		if (!fs::exists(root, ec)) {
			return;
		}

		for (const auto& entry : fs::directory_iterator(root, ec)) {
			const std::string dir = entry.path().string();
			if (read_line(dir + "/name").rfind("package-", 0) != 0) {
				continue;
			}
			if (!package.init(dir)) {
				continue;
			}
			for (const auto& sub : fs::directory_iterator(entry.path(), ec)) {
				if (read_line(sub.path().string() + "/name") == "core") {
					core.init(sub.path().string());
					break;
				}
			}
			break;
		}
	}

	void init_device() {
#if defined(TOPK_WITH_NVML)
		if (!g_device_requested) {
			return;
		}
		if (nvmlInit_v2() != NVML_SUCCESS) {
			return;
		}
		nvml_ready = true;

		const char* idx_env = std::getenv("TOPK_GPU_INDEX");
		const unsigned int index = idx_env != nullptr ? static_cast<unsigned int>(std::atoi(idx_env)) : 0u;
		if (nvmlDeviceGetHandleByIndex_v2(index, &nvml_device) != NVML_SUCCESS) {
			return;
		}

		unsigned int probe = 0;
		if (nvmlDeviceGetPowerUsage(nvml_device, &probe) != NVML_SUCCESS) {
			return;
		}

		device_ready = true;
		sampler_stop.store(false, std::memory_order_relaxed);
		sampler = std::thread([this]() { sample_loop(); });
#endif
	}

#if defined(TOPK_WITH_NVML)
	void sample_loop() {
		auto prev = std::chrono::steady_clock::now();
		unsigned int watts_mw = 0;
		while (!sampler_stop.load(std::memory_order_relaxed)) {
			if (nvmlDeviceGetPowerUsage(nvml_device, &watts_mw) == NVML_SUCCESS) {
				const auto now = std::chrono::steady_clock::now();
				const double dt = std::chrono::duration<double>(now - prev).count();
				prev = now;
				device_integral_j.store(device_integral_j.load(std::memory_order_relaxed) +
											static_cast<double>(watts_mw) * 1e-3 * dt,
										std::memory_order_relaxed);
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(kSamplePeriodMs));
		}
	}
#endif

	double device_joules() {
#if defined(TOPK_WITH_NVML)
		if (!device_ready) {
			return 0.0;
		}
		return device_integral_j.load(std::memory_order_relaxed);
#else
		return 0.0;
#endif
	}

	RaplDomain package;
	RaplDomain core;
	bool device_ready = false;
#if defined(TOPK_WITH_NVML)
	static constexpr int kSamplePeriodMs = 2;
	bool nvml_ready = false;
	nvmlDevice_t nvml_device{};
	std::atomic<double> device_integral_j{0.0};
	std::atomic<bool> sampler_stop{true};
	std::thread sampler;
#endif
};

constexpr std::size_t kChannels = 3;
Sample g_accumulator[kChannels];
double g_seconds[kChannels] = {0.0, 0.0, 0.0};
long g_count[kChannels] = {0, 0, 0};
int g_depth[kChannels] = {0, 0, 0};

std::size_t index_of(Channel channel) {
	if (channel == Channel::E2e) {
		return 0u;
	}
	return channel == Channel::Algo ? 1u : 2u;
}

} // namespace

void enable_device_counter() {
	g_device_requested = true;
}

Counter& counter() {
	static SystemCounter instance;
	return instance;
}

Scope::Scope(Channel ch) : start_seconds(0.0), channel(ch), active(g_depth[index_of(ch)] == 0) {
	++g_depth[index_of(ch)];
	if (active) {
		start = counter().read();
		start_seconds = now_seconds();
	}
}

Scope::~Scope() {
	close();
}

void Scope::close() {
	const std::size_t i = index_of(channel);
	if (g_depth[i] > 0) {
		--g_depth[i];
	}
	if (active) {
		g_accumulator[i] += counter().read() - start;
		g_seconds[i] += now_seconds() - start_seconds;
		++g_count[i];
		active = false;
	}
}

FullScope::FullScope() : start_seconds(0.0), active(g_depth[0] == 0 && g_depth[1] == 0) {
	++g_depth[0];
	++g_depth[1];
	if (active) {
		start = counter().read();
		start_seconds = now_seconds();
	}
}

FullScope::~FullScope() {
	close();
}

void FullScope::close() {
	for (std::size_t i = 0; i < 2; ++i) {
		if (g_depth[i] > 0) {
			--g_depth[i];
		}
	}
	if (active) {
		const Sample delta = counter().read() - start;
		const double seconds = now_seconds() - start_seconds;
		for (std::size_t i = 0; i < 2; ++i) {
			g_accumulator[i] += delta;
			g_seconds[i] += seconds;
			++g_count[i];
		}
		active = false;
	}
}

Sample take_accumulator(Channel channel) {
	const std::size_t i = index_of(channel);
	const Sample s = g_accumulator[i];
	g_accumulator[i] = Sample{};
	return s;
}

double take_seconds(Channel channel) {
	const std::size_t i = index_of(channel);
	const double s = g_seconds[i];
	g_seconds[i] = 0.0;
	return s;
}

long take_count(Channel channel) {
	const std::size_t i = index_of(channel);
	const long c = g_count[i];
	g_count[i] = 0;
	return c;
}

void reset_accumulators() {
	for (std::size_t i = 0; i < kChannels; ++i) {
		g_accumulator[i] = Sample{};
		g_seconds[i] = 0.0;
		g_count[i] = 0;
		g_depth[i] = 0;
	}
}

} // namespace common::energy
