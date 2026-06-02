import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.ir import *

def build_design():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            @device(AIEDevice.npu1_1col)
            def npu_device():
                memref_1024 = T.memref(1024, T.i32())
                memref_8 = T.memref(8, T.i32())

                ShimTile = tile(0, 0)
                ComputeTile = tile(0, 2)

                in_fifo = object_fifo("in_fifo", ShimTile, ComputeTile, 2, memref_1024)
                out_fifo = object_fifo("out_fifo", ComputeTile, ShimTile, 2, memref_1024)
                cfg_fifo = object_fifo("cfg_fifo", ShimTile, ComputeTile, 1, memref_8)

                bitonic_func = external_func(
                    "bitonic_step_kernel",
                    inputs=[memref_1024, memref_1024, memref_8],
                    link_with="bitonic.o"
                )

                @core(ComputeTile)
                def core_body():
                    for _ in for_(sys.maxsize):
                        elem_cfg = cfg_fifo.acquire(ObjectFifoPort.Consume, 1)
                        elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)

                        # Pass the function reference, NOT a string
                        call(bitonic_func, [elem_in, elem_out, elem_cfg])

                        in_fifo.release(ObjectFifoPort.Consume, 1)
                        out_fifo.release(ObjectFifoPort.Produce, 1)
                        cfg_fifo.release(ObjectFifoPort.Consume, 1)
                        yield_([])

                @runtime_sequence(memref_8, memref_1024, memref_1024)
                def seq(cfg, out, in_buf):
                    npu_dma_memcpy_nd(metadata="cfg_fifo", bd_id=3, mem=cfg, sizes=[1, 1, 1, 8], strides=[1, 1, 1, 1])
                    npu_dma_memcpy_nd(metadata="out_fifo", bd_id=4, mem=out, sizes=[1, 1, 4, 256], strides=[1, 1, 256, 1])
                    npu_dma_memcpy_nd(metadata="in_fifo", bd_id=5, mem=in_buf, sizes=[1, 1, 4, 256], strides=[1, 1, 256, 1])
                    npu_sync(column=0, row=0, direction=0, channel=0, column_num=1, row_num=1)

        return module

if __name__ == "__main__":
    print(build_design())
