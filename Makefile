BUILD_DIR := build
THESIS_DIR := doc/thesis

ARGS ?=

.PHONY: all help build-all build-cpu build-gpu build-npu clean \
	run-cpu run-gpu run-npu run-cases run-energy run-profiler \
	benchmark benchmark-cases benchmark-energy profiler-bootstrap profiler-clean \
	pin pin-show unpin lint specs a-test ab-test \
	thesis

all: build-all

help:
	@echo "Targets:"
	@echo "  build-all | build-cpu | build-gpu | build-npu"
	@echo "  run-cpu | run-gpu | run-npu       - run a backend binary (ARGS=...)"
	@echo ""
	@echo "  Measure (write test/prof/results/):"
	@echo "    run-cases                     - measure timing data via runner.py (ARGS=...)"
	@echo "    run-energy                    - measure energy data, all backends; sudo for RAPL (ARGS=...)"
	@echo "  Plot:"
	@echo "    run-profiler                  - plot results from measured data (ARGS=...)"
	@echo "  Pipelines (measure + plot):"
	@echo "    benchmark-cases               - timing: run-cases + run-profiler"
	@echo "    benchmark-energy              - energy: run-energy + run-profiler"
	@echo "    benchmark                     - everything: benchmark-cases + benchmark-energy"
	@echo ""
	@echo "  Thesis:"
	@echo "    thesis                        - compile the LaTeX thesis document"
	@echo ""
	@echo "  Conditions (sudo): pin | pin-show | unpin  - set/show/restore governor + GPU power (pin ARGS=watts)"
	@echo ""
	@echo "  profiler-bootstrap | profiler-clean | lint | specs | a-test | ab-test"

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
	@# default offload file: bitonic
	@if [ -z "$$NPU_OFFLOAD_XCLBIN" ]; then \
		export NPU_OFFLOAD_XCLBIN="$(BUILD_DIR)/NPU/bitonic.xclbin"; \
	fi; \
	./$(BUILD_DIR)/NPU/topk $(ARGS)

run-cases:
	./test/prof/run_cases.sh $(ARGS)

run-energy:
	./test/prof/energy/run_energy.sh $(ARGS)

run-profiler:
	./test/prof/profiler/run_profiler.sh $(ARGS)

profiler-bootstrap:
	./test/prof/profiler/bootstrap_env.sh

profiler-clean:
	./test/prof/clean_prof.sh

benchmark:
	$(MAKE) benchmark-cases
	$(MAKE) benchmark-energy

benchmark-cases:
	@echo "=== 1. Measuring timing (all backends) ==="
	-$(MAKE) run-cases
	@echo "=== 2. Generating timing plots ==="
	$(MAKE) run-profiler ARGS="--plot all --error-bars none --timing-csv-out ./test/prof/results/profile_cases.csv"

benchmark-energy:
	@echo "=== 1. Measuring energy (all backends) ==="
	-$(MAKE) run-energy
	@echo "=== 2. Generating energy plots ==="
	$(MAKE) run-profiler ARGS="--input ./test/prof/results/energy_output.json --plot energy-by-backend power-by-backend energy-vs-n power-vs-n edp-vs-n energy-per-element-vs-n time-vs-energy --energy-metric both --error-bars none --energy-csv-out ./test/prof/results/profile_energy.csv"

pin-show:
	ACTION=show ./scripts/pin_conditions.sh

pin:
	./scripts/pin_conditions.sh $(ARGS)

unpin:
	ACTION=restore ./scripts/pin_conditions.sh

lint:
	./scripts/lint.sh $(ARGS)

specs:
	./scripts/validate_specs.sh $(ARGS)

a-test:
	./scripts/a_test.sh $(ARGS)

ab-test:
	./scripts/ab_test.sh $(ARGS)

thesis:
	@echo "=== Compiling Thesis ==="
	cd $(THESIS_DIR) && latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=build main.tex
	cd $(THESIS_DIR) && cp build/main.pdf .

clean:
	rm -rf $(BUILD_DIR)
	-cd $(THESIS_DIR) && rm -rf build main.pdf
