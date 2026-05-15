BUILD_DIR := build

ARGS ?=

.PHONY: all help build-all build-cpu build-gpu build-npu clean \
	run-cpu run-gpu run-npu run-cases run-energy run-profiler \
	profiler-bootstrap profiler-clean smoke lint specs a-test ab-test

all: build-all

help:
	@echo "Targets:"
	@echo "  build-all | build-cpu | build-gpu | build-npu"
	@echo "  run-cpu | run-gpu | run-npu       - run backend binaries (ARGS=...)"
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
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=ON -DTOPK_BUILD_GPU=ON -DTOPK_BUILD_NPU=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-cpu: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=ON -DTOPK_BUILD_GPU=OFF -DTOPK_BUILD_NPU=OFF
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-gpu: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=OFF -DTOPK_BUILD_GPU=ON -DTOPK_BUILD_NPU=OFF
	cd $(BUILD_DIR) && make -j$(shell nproc)

build-npu: $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake .. -DTOPK_BUILD_CPU=OFF -DTOPK_BUILD_GPU=OFF -DTOPK_BUILD_NPU=ON
	cd $(BUILD_DIR) && make -j$(shell nproc)

run-cpu: build-cpu
	./$(BUILD_DIR)/CPU/topk $(ARGS)

run-gpu: build-gpu
	./$(BUILD_DIR)/GPU/topk $(ARGS)

run-npu: build-npu
	./$(BUILD_DIR)/NPU/topk $(ARGS)

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

benchmark:
	@echo "=== 0. Cleaning Previous Benchmark Artifacts ==="
	$(MAKE) profiler-clean
	@echo "=== 1. Running Heavy Benchmark Cases ==="
	-$(MAKE) run-cases ARGS="cpu gpu --types int uint float double fp16 --q-max 24 --min 0 --max 1000000000"
	@echo "=== 2. Generating Benchmark Plots ==="
	$(MAKE) run-profiler ARGS="--plot all --error-bars none"

lint:
	./scripts/lint.sh $(ARGS)

specs:
	./scripts/validate_specs.sh $(ARGS)

a-test:
	./scripts/a_test.sh $(ARGS)

ab-test:
	./scripts/ab_test.sh $(ARGS)

clean:
	rm -rf $(BUILD_DIR)
