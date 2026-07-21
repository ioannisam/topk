BUILD_DIR := build
THESIS_DIR := doc/thesis

ARGS ?=

.PHONY: all help build-all build-cpu build-gpu build-npu clean clean-build clean-results \
	run-cpu run-gpu run-npu measure-cases measure-energy measure-roofline measure-ncu plot \
	benchmark benchmark-cases benchmark-energy benchmark-roofline profiler-bootstrap \
	pin pin-show unpin lint specs a-test ab-test \
	thesis

all: build-all

help:
	@echo "Targets:"
	@echo "  build-all | build-cpu | build-gpu | build-npu"
	@echo "  run-cpu | run-gpu | run-npu       - run a backend binary (ARGS=...)"
	@echo ""
	@echo "  Measure (write bench/results/):"
	@echo "    measure-cases                 - measure timing data via runner.py (ARGS=...)"
	@echo "    measure-energy                - measure energy data, all backends; sudo for RAPL (ARGS=...)"
	@echo "    measure-roofline              - measure bandwidth ceilings, cache + transfer walls (ARGS=...)"
	@echo "    measure-ncu                   - GPU hardware counters via Nsight Compute (ARGS=...)"
	@echo "  Plot:"
	@echo "    plot                          - plot results from measured data (ARGS=...)"
	@echo "  Pipelines (measure + plot):"
	@echo "    benchmark-cases               - timing: measure-cases + plot"
	@echo "    benchmark-energy              - energy: measure-energy + plot"
	@echo "    benchmark-roofline            - roofline: measure-roofline + plot"
	@echo "    benchmark                     - everything: benchmark-cases + benchmark-energy + benchmark-roofline"
	@echo ""
	@echo "  Thesis:"
	@echo "    thesis                        - compile the LaTeX thesis document"
	@echo ""
	@echo "  Conditions (sudo): pin | pin-show | unpin  - set/show/restore governor + GPU power (pin ARGS=watts)"
	@echo ""
	@echo "  profiler-bootstrap | lint | specs | a-test | ab-test"
	@echo ""
	@echo "  Clean:"
	@echo "    clean-build                   - remove build/ and thesis build artifacts"
	@echo "    clean-results                 - remove bench/results/{raw,derived,plots} (measurement data!)"
	@echo "    clean                         - clean-build + clean-results"

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

measure-cases:
	./bench/run_cases.sh $(ARGS)

measure-energy:
	./bench/run_energy.sh $(ARGS)

measure-roofline:
	./bench/run_roofline.sh $(ARGS)

measure-ncu:
	./bench/lib/measure_ncu.sh --out bench/results/raw/ncu/report.txt -- ./build/GPU/topk $(ARGS)

plot:
	./bench/profiler/run_profiler.sh $(ARGS)

profiler-bootstrap:
	./bench/profiler/bootstrap_env.sh

benchmark:
	$(MAKE) benchmark-cases
	$(MAKE) benchmark-energy
	$(MAKE) benchmark-roofline

benchmark-roofline:
	@echo "=== 1. Measuring bandwidth ceilings and AI sweeps ==="
	-$(MAKE) measure-roofline
	@echo "=== 2. Generating roofline plots ==="
	$(MAKE) plot ARGS="--plot roofline memory-bandwidth-vs-n --error-bars none"

benchmark-cases:
	@echo "=== 1. Measuring timing (all backends) ==="
	-$(MAKE) measure-cases
	@echo "=== 2. Generating timing plots ==="
	$(MAKE) plot ARGS="--plot all --error-bars none"

benchmark-energy:
	@echo "=== 1. Measuring energy (all backends) ==="
	-$(MAKE) measure-energy
	@echo "=== 2. Generating energy plots ==="
	$(MAKE) plot ARGS="--input ./bench/results/raw/energy/output.json --plot energy-by-backend power-by-backend energy-vs-n power-vs-n edp-vs-n energy-per-element-vs-n time-vs-energy --energy-metric both --error-bars none"

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

clean-build:
	rm -rf $(BUILD_DIR)
	-cd $(THESIS_DIR) && rm -rf build main.pdf

clean-results:
	@if [ "$(FORCE)" = "1" ]; then \
		./bench/clean.sh; \
	else \
		printf "This deletes bench/results/{raw,derived,plots} - measurement data, not in git.\n"; \
		printf "Continue? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]|[yY][eE][sS]) ./bench/clean.sh ;; *) echo "Aborted." ;; esac; \
	fi

clean: clean-build clean-results
