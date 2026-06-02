OUT_DIR ?= build

ifndef VIRTUAL_ENV
$(error VIRTUAL_ENV is not set. Please source npu_venv/bin/activate first)
endif

PEANO_CLANG := $(VIRTUAL_ENV)/lib/python3.14/site-packages/llvm-aie/bin/clang++
MLIR_AIE_INC := $(VIRTUAL_ENV)/lib/python3.14/site-packages/mlir_aie/include

AIE_CXXFLAGS := -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses \
                -Wno-attributes -Wno-macro-redefined -DNDEBUG -I $(MLIR_AIE_INC)

hardware: $(OUT_DIR)/bitonic.xclbin $(OUT_DIR)/map_reduce.xclbin

$(OUT_DIR)/%.o: src/%.cc
	@mkdir -p $(OUT_DIR)
	$(PEANO_CLANG) $(AIE_CXXFLAGS) -c $< -o $@

$(OUT_DIR)/%.mlir: mlir/%.py
	@mkdir -p $(OUT_DIR)
	python3 $< -d npu > $@

$(OUT_DIR)/%.xclbin: $(OUT_DIR)/%.mlir $(OUT_DIR)/%.o
	cd $(OUT_DIR) && aiecc --aie-generate-xclbin --aie-generate-npu-insts \
		--no-xchesscc --no-xbridge \
		--xclbin-name=$*.xclbin --npu-insts-name=$*.bin $(notdir $<)

clean:
	rm -rf $(OUT_DIR)
