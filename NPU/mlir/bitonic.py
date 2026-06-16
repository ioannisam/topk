import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.ir import *

NUM_COLS = 4
TILE = 1024
CHUNKS_PER_COL = 256
BATCH_CHUNKS = NUM_COLS * CHUNKS_PER_COL
ELEMS_PER_COL = CHUNKS_PER_COL * TILE
BATCH_ELEMS = BATCH_CHUNKS * TILE

def build_design():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            @device(AIEDevice.npu1)
            def npu_device():
                memref_tile = T.memref(TILE, T.i32())
                memref_batch = T.memref(BATCH_ELEMS, T.i32())

                sort_func = external_func("bitonic_sort_runs", inputs=[memref_tile, memref_tile], link_with="bitonic.o")

                tiles = {}
                fifos = {}
                for col in range(NUM_COLS):
                    tiles[col] = {"shim": tile(col, 0), "compute": tile(col, 2)}
                    fifos[col] = {
                        "in": object_fifo(f"in_{col}", tiles[col]["shim"], tiles[col]["compute"], 2, memref_tile),
                        "out": object_fifo(f"out_{col}", tiles[col]["compute"], tiles[col]["shim"], 2, memref_tile),
                    }

                def build_core(compute_tile, in_f, out_f):
                    @core(compute_tile)
                    def core_body():
                        for _ in for_(sys.maxsize):
                            elem_in = in_f.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = out_f.acquire(ObjectFifoPort.Produce, 1)
                            call(sort_func, [elem_in, elem_out])
                            in_f.release(ObjectFifoPort.Consume, 1)
                            out_f.release(ObjectFifoPort.Produce, 1)
                            yield_([])

                for col in range(NUM_COLS):
                    build_core(tiles[col]["compute"], fifos[col]["in"], fifos[col]["out"])

                @runtime_sequence(memref_batch, memref_batch)
                def seq(out, inp):
                    for col in range(NUM_COLS):
                        elem_offset = col * ELEMS_PER_COL
                        bd_base = col * 2
                        npu_dma_memcpy_nd(metadata=f"out_{col}", bd_id=bd_base + 0, mem=out,
                                          offsets=[0, 0, 0, elem_offset],
                                          sizes=[1, 1, CHUNKS_PER_COL, TILE], strides=[1, 1, TILE, 1])
                        npu_dma_memcpy_nd(metadata=f"in_{col}", bd_id=bd_base + 1, mem=inp,
                                          offsets=[0, 0, 0, elem_offset],
                                          sizes=[1, 1, CHUNKS_PER_COL, TILE], strides=[1, 1, TILE, 1])

                    npu_sync(column=0, row=0, direction=0, channel=0, column_num=NUM_COLS, row_num=1)

        return module

if __name__ == "__main__":
    print(build_design())
