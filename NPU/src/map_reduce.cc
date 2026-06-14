#include <aie_api/aie.hpp>
#include <cstdint>

extern "C" {

void map_reduce_step_kernel(int32_t* restrict in_buf, int32_t* restrict out_buf, int32_t* restrict cfg_buf) {
    int32_t threshold = cfg_buf[0];
    int32_t want_max  = cfg_buf[1]; 

    constexpr int vector_width = 16;
    constexpr int total_elements = 1024;
    
    aie::vector<int32_t, vector_width> t_vec = aie::broadcast<int32_t, vector_width>(threshold);
    aie::vector<int32_t, vector_width> sentinel_vec = aie::broadcast<int32_t, vector_width>(cfg_buf[2]);

    int32_t valid_count = 0;
    int32_t* out_data_ptr = out_buf + 1; 

    for (int i = 0; i < total_elements; i += vector_width) {
        aie::vector<int32_t, vector_width> v = aie::load_v<vector_width>(in_buf + i);
        aie::mask<vector_width> cmp_mask = want_max ? aie::gt(v, t_vec) : aie::lt(v, t_vec);
        
        uint32_t mask_bits = cmp_mask.to_uint32();
        
        // cap at 1008 to ensure we don't overflow the 1024-element out_buf
        if (__builtin_expect(mask_bits != 0, 0) && valid_count < 1008) {
            aie::vector<int32_t, vector_width> out_v = aie::select(sentinel_vec, v, cmp_mask);
            aie::store_v(out_data_ptr, out_v);
            out_data_ptr += vector_width;
            valid_count += vector_width;
        }
    }
    
    out_buf[0] = valid_count;
}

} // extern "C"
