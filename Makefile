BUILD_DIR := build

ARGS ?=

.PHONY: all help build-all build-cpu build-gpu build-npu build-gt clean \
	run-cases run-energy run-profiler profiler-bootstrap profiler-clean \
	smoke lint specs a-test ab-test

all: build-all

help:
	@echo "Targets:"
	@echo "  build-all | build-cpu | build-gpu | build-npu | build-gt"
	@echo "  smoke                           - run smoke tests"
	@echo "  run-cases                       - run testcase suite via runner.py (ARGS=...)"
	@echo "  run-energy                      - energy measurement run (ARGS=...)"
	@echo "  run-profiler                    - run Python profiler to generate plots (ARGS=...)"
	@echo "  profiler-bootstrap              - create profiler venv"
	@echo "  profiler-clean                  - clean profiler artifacts"
	@echo "  lint | specs | a-test | ab-test"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

build-all: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=ON -DTOPK_BUILD_GPU=ON -DTOPK_BUILD_NPU=ON -DTOPK_BUILD_GT=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-cpu: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=ON -DTOPK_BUILD_GPU=OFF -DTOPK_BUILD_NPU=OFF -DTOPK_BUILD_GT=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-gpu: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=OFF -DTOPK_BUILD_GPU=ON -DTOPK_BUILD_NPU=OFF -DTOPK_BUILD_GT=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-npu: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=OFF -DTOPK_BUILD_GPU=OFF -DTOPK_BUILD_NPU=ON -DTOPK_BUILD_GT=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-gt: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=OFF -DTOPK_BUILD_GPU=OFF -DTOPK_BUILD_NPU=OFF -DTOPK_BUILD_GT=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

smoke:
	./test/smoke/run_smoke_tests.sh

run-cases:
	python3 ./test/prof/runner.py $(ARGS)

run-energy:
	./test/prof/energy/run_energy_profile.sh $(ARGS)

run-profiler:
	./test/prof/profiler/run_profiler.sh $(ARGS)

profiler-bootstrap:
	./test/prof/profiler/bootstrap_env.sh

profiler-clean:
	./test/prof/clean_prof.sh

lint:
	./scripts/lint_cpp.sh $(ARGS)

specs:
	./scripts/validate_specs.sh $(ARGS)

a-test:
	./scripts/a_test.sh $(ARGS)

ab-test:
	./scripts/ab_test.sh $(ARGS)

clean:
	rm -rf $(BUILD_DIR)
