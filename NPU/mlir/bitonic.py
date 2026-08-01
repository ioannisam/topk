import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.ir import *

NUM_COLS = 4
CORES_PER_COL = 2
TILE = 1024
CHUNKS_PER_CORE = 128
ELEMS_PER_CORE = CHUNKS_PER_CORE * TILE
ELEMS_PER_COL = CORES_PER_COL * ELEMS_PER_CORE
BATCH_ELEMS = NUM_COLS * ELEMS_PER_COL


def build_design():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            @device(AIEDevice.npu1)
            def npu_device():
                memref_tile = T.memref(TILE, T.i32())
                memref_batch = T.memref(BATCH_ELEMS, T.i32())

                sort_func = external_func("bitonic_sort_runs", inputs=[memref_tile, memref_tile], link_with="bitonic.o")

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

                comps = {}
                fifos = {}
                for col in range(NUM_COLS):
                    shim = tile(col, 0)
                    for r in range(CORES_PER_COL):
                        comp = tile(col, 2 + r)
                        comps[(col, r)] = comp
                        fifos[(col, r)] = (
                            object_fifo(f"in_{col}_{r}", shim, comp, 2, memref_tile),
                            object_fifo(f"out_{col}_{r}", comp, shim, 2, memref_tile),
                        )

                for col in range(NUM_COLS):
                    for r in range(CORES_PER_COL):
                        in_f, out_f = fifos[(col, r)]
                        build_core(comps[(col, r)], in_f, out_f)

                @runtime_sequence(memref_batch, memref_batch)
                def seq(out, inp):
                    for col in range(NUM_COLS):
                        for r in range(CORES_PER_COL):
                            elem_offset = col * ELEMS_PER_COL + r * ELEMS_PER_CORE
                            bd_base = (col * CORES_PER_COL + r) * 2
                            npu_dma_memcpy_nd(
                                metadata=f"out_{col}_{r}",
                                bd_id=bd_base + 0,
                                mem=out,
                                offsets=[0, 0, 0, elem_offset],
                                sizes=[1, 1, CHUNKS_PER_CORE, TILE],
                                strides=[1, 1, TILE, 1],
                            )
                            npu_dma_memcpy_nd(
                                metadata=f"in_{col}_{r}",
                                bd_id=bd_base + 1,
                                mem=inp,
                                offsets=[0, 0, 0, elem_offset],
                                sizes=[1, 1, CHUNKS_PER_CORE, TILE],
                                strides=[1, 1, TILE, 1],
                            )

                    npu_sync(column=0, row=0, direction=0, channel=0, column_num=NUM_COLS, row_num=1)

        return module


if __name__ == "__main__":
    print(build_design())
