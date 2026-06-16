import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.ir import *

NUM_COLS = 4
NUM_STAGES = 4
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

                stage_funcs = [
                    external_func(f"pipeline_core_{s + 1}", inputs=[memref_tile, memref_tile], link_with="bitonic.o")
                    for s in range(NUM_STAGES)
                ]

                cols = {}
                for col in range(NUM_COLS):
                    shim = tile(col, 0)
                    compute = [tile(col, 2 + s) for s in range(NUM_STAGES)]

                    in_fifo = object_fifo(f"in_{col}", shim, compute[0], 2, memref_tile)
                    out_fifo = object_fifo(f"out_{col}", compute[NUM_STAGES - 1], shim, 2, memref_tile)
                    links = [
                        object_fifo(f"link_{col}_{s}", compute[s], compute[s + 1], 2, memref_tile)
                        for s in range(NUM_STAGES - 1)
                    ]
                    cols[col] = {"compute": compute, "in": in_fifo, "out": out_fifo, "links": links}

                def build_stage(compute_tile, func, in_f, out_f):
                    @core(compute_tile)
                    def core_body():
                        for _ in for_(sys.maxsize):
                            elem_in = in_f.acquire(ObjectFifoPort.Consume, 1)
                            elem_out = out_f.acquire(ObjectFifoPort.Produce, 1)
                            call(func, [elem_in, elem_out])
                            in_f.release(ObjectFifoPort.Consume, 1)
                            out_f.release(ObjectFifoPort.Produce, 1)
                            yield_([])

                for col in range(NUM_COLS):
                    c = cols[col]
                    for s in range(NUM_STAGES):
                        in_f = c["in"] if s == 0 else c["links"][s - 1]
                        out_f = c["out"] if s == NUM_STAGES - 1 else c["links"][s]
                        build_stage(c["compute"][s], stage_funcs[s], in_f, out_f)

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
