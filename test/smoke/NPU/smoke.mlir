module @smoke {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)

    %core_0_2 = aie.core(%tile_0_2) {
      aie.end
    }
  }
}
