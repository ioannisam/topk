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
                ShimTile = tile(0, 0)
                Core1 = tile(0, 2)
                Core2 = tile(0, 3)
                Core3 = tile(0, 4)
                Core4 = tile(0, 5)

                in_fifo    = object_fifo("in_fifo", ShimTile, Core1, 2, memref_1024)
                stream_1_2 = object_fifo("stream_1_2", Core1, Core2, 2, memref_1024)
                stream_2_3 = object_fifo("stream_2_3", Core2, Core3, 2, memref_1024)
                stream_3_4 = object_fifo("stream_3_4", Core3, Core4, 2, memref_1024)
                out_fifo   = object_fifo("out_fifo", Core4, ShimTile, 2, memref_1024)

                stage_1 = external_func("pipeline_core_1", inputs=[memref_1024, memref_1024], link_with="bitonic.o")
                stage_2 = external_func("pipeline_core_2", inputs=[memref_1024, memref_1024], link_with="bitonic.o")
                stage_3 = external_func("pipeline_core_3", inputs=[memref_1024, memref_1024], link_with="bitonic.o")
                stage_4 = external_func("pipeline_core_4", inputs=[memref_1024, memref_1024], link_with="bitonic.o")

                @core(Core1)
                def core1_body():
                    for _ in for_(sys.maxsize):
                        elem_in = in_fifo.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = stream_1_2.acquire(ObjectFifoPort.Produce, 1)
                        call(stage_1, [elem_in, elem_out])
                        in_fifo.release(ObjectFifoPort.Consume, 1)
                        stream_1_2.release(ObjectFifoPort.Produce, 1)
                        yield_([])

                @core(Core2)
                def core2_body():
                    for _ in for_(sys.maxsize):
                        elem_in = stream_1_2.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = stream_2_3.acquire(ObjectFifoPort.Produce, 1)
                        call(stage_2, [elem_in, elem_out])
                        stream_1_2.release(ObjectFifoPort.Consume, 1)
                        stream_2_3.release(ObjectFifoPort.Produce, 1)
                        yield_([])

                @core(Core3)
                def core3_body():
                    for _ in for_(sys.maxsize):
                        elem_in = stream_2_3.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = stream_3_4.acquire(ObjectFifoPort.Produce, 1)
                        call(stage_3, [elem_in, elem_out])
                        stream_2_3.release(ObjectFifoPort.Consume, 1)
                        stream_3_4.release(ObjectFifoPort.Produce, 1)
                        yield_([])

                @core(Core4)
                def core4_body():
                    for _ in for_(sys.maxsize):
                        elem_in = stream_3_4.acquire(ObjectFifoPort.Consume, 1)
                        elem_out = out_fifo.acquire(ObjectFifoPort.Produce, 1)
                        call(stage_4, [elem_in, elem_out])
                        stream_3_4.release(ObjectFifoPort.Consume, 1)
                        out_fifo.release(ObjectFifoPort.Produce, 1)
                        yield_([])

                @runtime_sequence(memref_1024, memref_1024)
                def seq(in_buf, out_buf):
                    npu_dma_memcpy_nd(metadata="in_fifo", bd_id=0, mem=in_buf, sizes=[1, 1, 4, 256], strides=[1, 1, 256, 1])
                    npu_dma_memcpy_nd(metadata="out_fifo", bd_id=1, mem=out_buf, sizes=[1, 1, 4, 256], strides=[1, 1, 256, 1])
                    npu_sync(column=0, row=0, direction=0, channel=0, column_num=1, row_num=1)

        return module

if __name__ == "__main__":
    print(build_design())
