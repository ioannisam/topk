import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.ir import *

def build_design():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            @device(AIEDevice.npu1)
            def npu_device():
                memref_1024 = T.memref(1024, T.i32())
                memref_4 = T.memref(4, T.i32())
                
                memref_batch = T.memref(262144, T.i32())   # 65536 * 4 columns
                memref_cfg_batch = T.memref(1024, T.i32()) # 256 * 4 columns

                map_reduce_func = external_func(
                    "map_reduce_step_kernel",
                    inputs=[memref_1024, memref_1024, memref_4],
                    link_with="map_reduce.o"
                )

                NUM_COLS = 4
                
                tiles = {}
                for col in range(NUM_COLS):
                    tiles[col] = {
                        "shim": tile(col, 0),
                        "compute": tile(col, 2)
                    }

                fifos = {}
                for col in range(NUM_COLS):
                    fifos[col] = {
                        "in": object_fifo(f"in_fifo_{col}", tiles[col]["shim"], tiles[col]["compute"], 2, memref_1024),
                        "out": object_fifo(f"out_candidates_{col}", tiles[col]["compute"], tiles[col]["shim"], 2, memref_1024),
                        "cfg": object_fifo(f"cfg_fifo_{col}", tiles[col]["shim"], tiles[col]["compute"], 2, memref_4)
                    }

                def build_core(compute_tile, in_f, out_f, cfg_f):
                    @core(compute_tile)
                    def core_body():
                        for _ in for_(sys.maxsize):
                            elem_cfg = cfg_f.acquire(ObjectFifoPort.Consume, 1)
                            elem_in = in_f.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = out_f.acquire(ObjectFifoPort.Produce, 1)

                            call(map_reduce_func, [elem_in, elem_out, elem_cfg])

                            in_f.release(ObjectFifoPort.Consume, 1)
                            out_f.release(ObjectFifoPort.Produce, 1)
                            cfg_f.release(ObjectFifoPort.Consume, 1)
                            yield_([])

                for col in range(NUM_COLS):
                    build_core(tiles[col]["compute"], fifos[col]["in"], fifos[col]["out"], fifos[col]["cfg"])

                @runtime_sequence(memref_cfg_batch, memref_batch, memref_batch)
                def seq(cfg, out, inp):
                    # Column 0
                    npu_dma_memcpy_nd(metadata="cfg_fifo_0", bd_id=0, mem=cfg, offsets=[0, 0, 0, 0], sizes=[1, 1, 64, 4], strides=[1, 1, 4, 1])
                    npu_dma_memcpy_nd(metadata="out_candidates_0", bd_id=1, mem=out, offsets=[0, 0, 0, 0], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])
                    npu_dma_memcpy_nd(metadata="in_fifo_0", bd_id=2, mem=inp, offsets=[0, 0, 0, 0], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])

                    # Column 1 (Offset by 1/4 of total elements)
                    npu_dma_memcpy_nd(metadata="cfg_fifo_1", bd_id=3, mem=cfg, offsets=[0, 0, 0, 256], sizes=[1, 1, 64, 4], strides=[1, 1, 4, 1])
                    npu_dma_memcpy_nd(metadata="out_candidates_1", bd_id=4, mem=out, offsets=[0, 0, 0, 65536], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])
                    npu_dma_memcpy_nd(metadata="in_fifo_1", bd_id=5, mem=inp, offsets=[0, 0, 0, 65536], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])

                    # Column 2 (Offset by 2/4 of total elements)
                    npu_dma_memcpy_nd(metadata="cfg_fifo_2", bd_id=6, mem=cfg, offsets=[0, 0, 0, 512], sizes=[1, 1, 64, 4], strides=[1, 1, 4, 1])
                    npu_dma_memcpy_nd(metadata="out_candidates_2", bd_id=7, mem=out, offsets=[0, 0, 0, 131072], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])
                    npu_dma_memcpy_nd(metadata="in_fifo_2", bd_id=8, mem=inp, offsets=[0, 0, 0, 131072], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])

                    # Column 3 (Offset by 3/4 of total elements)
                    npu_dma_memcpy_nd(metadata="cfg_fifo_3", bd_id=9, mem=cfg, offsets=[0, 0, 0, 768], sizes=[1, 1, 64, 4], strides=[1, 1, 4, 1])
                    npu_dma_memcpy_nd(metadata="out_candidates_3", bd_id=10, mem=out, offsets=[0, 0, 0, 196608], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])
                    npu_dma_memcpy_nd(metadata="in_fifo_3", bd_id=11, mem=inp, offsets=[0, 0, 0, 196608], sizes=[1, 1, 1024, 64], strides=[1, 1, 64, 1])

                    # Wait for all 4 channels to finish
                    npu_sync(column=0, row=0, direction=0, channel=0, column_num=4, row_num=1)

        return module

if __name__ == "__main__":
    print(build_design())
