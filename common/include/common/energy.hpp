#pragma once

#include <string>

namespace common::energy {

struct Sample {
	double package_j = 0.0;
	double core_j = 0.0;
	double device_j = 0.0;

	double total() const {
		return package_j + device_j;
	}
};

inline Sample operator-(const Sample& lhs, const Sample& rhs) {
	return Sample{lhs.package_j - rhs.package_j, lhs.core_j - rhs.core_j, lhs.device_j - rhs.device_j};
}

inline Sample& operator+=(Sample& lhs, const Sample& rhs) {
	lhs.package_j += rhs.package_j;
	lhs.core_j += rhs.core_j;
	lhs.device_j += rhs.device_j;
	return lhs;
}

struct Summary {
	Sample e2e_total;
	Sample algo_total;
	Sample loop_total;
	Sample wait_total;
	double e2e_seconds = 0.0;
	double algo_seconds = 0.0;
	double loop_seconds = 0.0;
	double wait_seconds = 0.0;
	long wait_count = 0;
	int iterations = 0;
	bool available = false;
};

class Counter {
  public:
	virtual ~Counter() = default;

	virtual bool available() const = 0;
	virtual Sample read() = 0;
	virtual std::string describe() const = 0;
};

void enable_device_counter();

Counter& counter();

enum class Channel { E2e, Algo, Wait };

class Scope {
  public:
	explicit Scope(Channel channel);
	~Scope();

	void close();

	Scope(const Scope&) = delete;
	Scope& operator=(const Scope&) = delete;

  private:
	Sample start;
	double start_seconds;
	Channel channel;
	bool active;
	bool closed = false;
};

class FullScope {
  public:
	FullScope();
	~FullScope();

	void close();

	FullScope(const FullScope&) = delete;
	FullScope& operator=(const FullScope&) = delete;

  private:
	Sample start;
	double start_seconds;
	bool active;
	bool closed = false;
};

Sample take_accumulator(Channel channel);
double take_seconds(Channel channel);
long take_count(Channel channel);
void reset_accumulators();

} // namespace common::energy
