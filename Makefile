BUILD_DIR := build
THESIS_DIR := doc/thesis

ARGS ?=
GPU_W ?= 50
NPU_VENV ?= npu_venv

CATALOGUE_DIST ?= uniform
CATALOGUE_K ?= 8 131072
CATALOGUE_DTYPES ?= int uint float double half
CATALOGUE_METRIC ?= net
CATALOGUE := --dist $(CATALOGUE_DIST) --fanout-k $(CATALOGUE_K) --dtype $(CATALOGUE_DTYPES) --energy-metric $(CATALOGUE_METRIC)

PLOT_CASES := --plot all --error-bars none --energy-csv-out '' $(CATALOGUE)
PLOT_ENERGY := --input ./bench/results/raw/energy/output.json --plot energy-by-backend power-by-backend energy-vs-n power-vs-n edp-vs-n energy-per-element-vs-n time-vs-energy --error-bars none --timing-csv-out '' $(CATALOGUE)
PLOT_ROOFLINE := --plot roofline roofline-kernels memory-bandwidth-vs-n roof-utilization --error-bars none --timing-csv-out '' --energy-csv-out '' $(CATALOGUE)
PLOT_DISTS := --input ./bench/results/raw/dists/output.json --plot dist-compare --fanout-k $(CATALOGUE_K) --dtype float half --error-bars none --timing-csv-out ./bench/results/derived/dists.csv --energy-csv-out ''

.PHONY: all help build-all build-cpu build-gpu build-npu verify-artifacts \
	clean clean-build clean-results \
	run-cpu run-gpu run-npu measure-cases measure-energy measure-dists measure-roofline measure-ncu plot \
	benchmark benchmark-cases benchmark-energy benchmark-dists benchmark-roofline profiler-bootstrap \
	pin pin-show unpin lint specs a-test ab-test \
	thesis thesis-greek thesis-english

all: build-all

help:
	@echo "Targets:"
	@echo "  build-all | build-cpu | build-gpu | build-npu"
	@echo "  run-cpu | run-gpu | run-npu       - run a backend binary (ARGS=...)"
	@echo ""
	@echo "  Measure (write bench/results/):"
	@echo "    measure-cases                 - measure timing data via runner.py (ARGS=...)"
	@echo "    measure-energy                - measure energy data, all backends; sudo for RAPL (ARGS=...)"
	@echo "    measure-dists                 - measure input-distribution sensitivity on a reduced grid (ARGS=...)"
	@echo "    measure-roofline              - measure bandwidth + compare-exchange roofs, cache + transfer walls (ARGS=...)"
	@echo "    measure-ncu                   - GPU hardware counters via Nsight Compute (ARGS=...)"
	@echo "  Plot:"
	@echo "    plot                          - plot results from measured data (ARGS=...)"
	@echo "  Pipelines (measure + plot):"
	@echo "    benchmark-cases               - timing: measure-cases + plot"
	@echo "    benchmark-energy              - energy: measure-energy + plot"
	@echo "    benchmark-dists               - distributions: measure-dists + plot"
	@echo "    benchmark-roofline            - roofline: measure-roofline + plot"
	@echo "    benchmark                     - UNATTENDED full run: clean, build, pin, measure x3,"
	@echo "                                    plot x3, unpin. Asks for sudo once up front, then"
	@echo "                                    needs no further input. GPU cap via GPU_W=50."
	@echo ""
	@echo "  Thesis:"
	@echo "    thesis                        - compile both thesis-greek and thesis-english"
	@echo "    thesis-greek                  - compile the Greek LaTeX thesis document"
	@echo "    thesis-english                - compile the English LaTeX thesis document"
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

verify-artifacts:
	@# runner.py silently drops an NPU algo whose xclbin is missing, so fail loudly here.
	@for x in bitonic map_reduce; do \
		if [ ! -f "$(BUILD_DIR)/NPU/$$x.xclbin" ]; then \
			echo "error: $(BUILD_DIR)/NPU/$$x.xclbin missing - NPU $$x would be skipped silently" >&2; exit 1; \
		fi; \
		echo "  $(BUILD_DIR)/NPU/$$x.xclbin ($$(stat -c%s "$(BUILD_DIR)/NPU/$$x.xclbin") bytes)"; \
	done
	@for b in CPU GPU NPU; do \
		if [ ! -x "$(BUILD_DIR)/$$b/topk" ]; then echo "error: $(BUILD_DIR)/$$b/topk missing" >&2; exit 1; fi; \
	done

run-cpu: build-cpu
	./$(BUILD_DIR)/CPU/topk $(ARGS)

run-gpu: build-gpu
	./$(BUILD_DIR)/GPU/topk $(ARGS)

run-npu: build-npu
	@# offload file follows algo= in ARGS; falls back to bitonic when there is no matching xclbin
	@algo=$$(printf '%s\n' $(ARGS) | sed -n 's/^algo=//p' | tail -1); \
	if [ -z "$$algo" ] || [ ! -f "$(BUILD_DIR)/NPU/$$algo.xclbin" ]; then algo=bitonic; fi; \
	if [ -z "$$NPU_OFFLOAD_XCLBIN" ]; then \
		export NPU_OFFLOAD_XCLBIN="$(BUILD_DIR)/NPU/$$algo.xclbin"; \
	fi; \
	./$(BUILD_DIR)/NPU/topk $(ARGS)

measure-cases:
	./bench/measure/run_cases.sh $(ARGS)

measure-energy:
	./bench/measure/run_energy.sh $(ARGS)

measure-dists:
	./bench/measure/run_dists.sh $(ARGS)

measure-roofline:
	./bench/measure/run_roofline.sh $(ARGS)

measure-ncu:
	./bench/lib/measure_ncu.sh --out bench/results/raw/ncu/report.txt -- ./build/GPU/topk $(ARGS)

plot:
	./bench/profiler/run_profiler.sh $(ARGS)

profiler-bootstrap:
	./bench/profiler/bootstrap_env.sh

benchmark:
	@command -v sudo >/dev/null 2>&1 || { echo "error: sudo is required (pin/unpin + RAPL)" >&2; exit 1; }
	@test -f ./$(NPU_VENV)/bin/activate || { echo "error: $(NPU_VENV) not found (needed for the xclbin build)" >&2; exit 1; }
	@echo "This deletes bench/results/ and rebuilds everything, then runs the full sweep."
	@echo "Enter your password once below; the rest of the run is unattended."
	@sudo -v
	@set -u; \
	( while true; do sudo -n true 2>/dev/null; sleep 50; kill -0 $$$$ 2>/dev/null || exit 0; done ) & \
	keepalive=$$!; pinned=0; \
	trap 'rc=$$?; echo "=== Restoring conditions ==="; \
	      if [ "$$pinned" = "1" ]; then ACTION=restore ./scripts/pin_conditions.sh; fi; \
	      kill $$keepalive 2>/dev/null; \
	      echo "=== benchmark finished (exit $$rc) ==="; exit $$rc' EXIT INT TERM; \
	echo "=== 1/6 Clean ==="; \
	$(MAKE) clean-build && $(MAKE) clean-results FORCE=1 || exit 1; \
	echo "=== 2/6 Build ==="; \
	$(MAKE) build-all && $(MAKE) verify-artifacts || exit 1; \
	echo "=== 3/6 Pin conditions (GPU $(GPU_W) W) ==="; \
	./scripts/pin_conditions.sh $(GPU_W) && pinned=1; \
	sudo -n chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj \
		/sys/class/powercap/intel-rapl:*/*/energy_uj 2>/dev/null || true; \
	echo "=== 4/6 Measure ==="; \
	$(MAKE) measure-cases    || echo "WARNING: measure-cases failed"; \
	$(MAKE) measure-energy   || echo "WARNING: measure-energy failed"; \
	$(MAKE) measure-dists    || echo "WARNING: measure-dists failed"; \
	$(MAKE) measure-roofline || echo "WARNING: measure-roofline failed"; \
	echo "=== 5/6 Plot ==="; \
	$(MAKE) plot ARGS="$(PLOT_CASES)"    || echo "WARNING: timing plots failed"; \
	$(MAKE) plot ARGS="$(PLOT_ENERGY)"   || echo "WARNING: energy plots failed"; \
	$(MAKE) plot ARGS="$(PLOT_DISTS)"    || echo "WARNING: distribution plots failed"; \
	$(MAKE) plot ARGS="$(PLOT_ROOFLINE)" || echo "WARNING: roofline plots failed"; \
	echo "=== 6/6 Done ==="

benchmark-roofline:
	@echo "=== 1. Measuring bandwidth ceilings and AI sweeps ==="
	-$(MAKE) measure-roofline
	@echo "=== 2. Generating roofline plots ==="
	$(MAKE) plot ARGS="$(PLOT_ROOFLINE)"

benchmark-cases:
	@echo "=== 1. Measuring timing (all backends) ==="
	-$(MAKE) measure-cases
	@echo "=== 2. Generating timing plots ==="
	$(MAKE) plot ARGS="$(PLOT_CASES)"

benchmark-energy:
	@echo "=== 1. Measuring energy (all backends) ==="
	-$(MAKE) measure-energy
	@echo "=== 2. Generating energy plots ==="
	$(MAKE) plot ARGS="$(PLOT_ENERGY)"

benchmark-dists:
	@echo "=== 1. Measuring distribution sensitivity (reduced grid) ==="
	-$(MAKE) measure-dists
	@echo "=== 2. Generating distribution plots ==="
	$(MAKE) plot ARGS="$(PLOT_DISTS)"

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

thesis: thesis-greek thesis-english

thesis-greek:
	@echo "=== Compiling Thesis (Greek) ==="
	cd $(THESIS_DIR)/greek && mkdir -p build/frontmatter build/chapters build/appendices
	cd $(THESIS_DIR)/greek && latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=build main.tex
	cd $(THESIS_DIR)/greek && cp build/main.pdf ../thesis-greek.pdf

thesis-english:
	@echo "=== Compiling Thesis (English) ==="
	cd $(THESIS_DIR)/english && mkdir -p build/frontmatter build/chapters build/appendices
	cd $(THESIS_DIR)/english && latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=build main.tex
	cd $(THESIS_DIR)/english && cp build/main.pdf ../thesis-english.pdf

clean-build:
	rm -rf $(BUILD_DIR)
	-cd $(THESIS_DIR) && rm -rf greek/build english/build thesis-greek.pdf thesis-english.pdf

clean-results:
	@if [ "$(FORCE)" = "1" ]; then \
		./bench/clean.sh; \
	else \
		printf "This deletes bench/results/{raw,derived,plots} - tracked in git, but any regenerated or uncommitted data will be lost.\n"; \
		printf "Continue? [y/N] "; \
		read ans; \
		case "$$ans" in [yY]|[yY][eE][sS]) ./bench/clean.sh ;; *) echo "Aborted." ;; esac; \
	fi

clean: clean-build clean-results
