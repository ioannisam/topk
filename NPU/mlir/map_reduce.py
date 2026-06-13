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
                
                memref_batch = T.memref(1048576, T.i32())
                memref_cfg_batch = T.memref(4096, T.i32())

                map_reduce_func = external_func(
                    "map_reduce_step_kernel",
                    inputs=[memref_1024, memref_1024, memref_4],
                    link_with="map_reduce.o"
                )

                NUM_COLS = 4 # hardware exposes only 4 columns
                
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
                    # 1,048,576 total elements / 4 columns = 262,144 elements per column.
                    # 262,144 elements / 64 vector size = 4,096 DMA chunk cycles per column.
                    
                    for col in range(NUM_COLS):
                        elem_offset = col * 262144
                        cfg_offset = col * 1024  # 256 chunks per col * 4 ints
                        bd_base = col * 3
                        
                        npu_dma_memcpy_nd(metadata=f"cfg_fifo_{col}", bd_id=bd_base+0, mem=cfg, offsets=[0, 0, 0, cfg_offset], sizes=[1, 1, 256, 4], strides=[1, 1, 4, 1])
                        npu_dma_memcpy_nd(metadata=f"out_candidates_{col}", bd_id=bd_base+1, mem=out, offsets=[0, 0, 0, elem_offset], sizes=[1, 1, 4096, 64], strides=[1, 1, 64, 1])
                        npu_dma_memcpy_nd(metadata=f"in_fifo_{col}", bd_id=bd_base+2, mem=inp, offsets=[0, 0, 0, elem_offset], sizes=[1, 1, 4096, 64], strides=[1, 1, 64, 1])

                    # Wait for all 4 channels to finish
                    npu_sync(column=0, row=0, direction=0, channel=0, column_num=NUM_COLS, row_num=1)

        return module

if __name__ == "__main__":
    print(build_design())
