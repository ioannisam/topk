#include <aie_api/aie.hpp>
#include <cstdint>

extern "C" {

void map_reduce_step_kernel(int32_t* restrict in_buf, int32_t* restrict out_buf, int32_t* restrict cfg_buf) {
    int32_t threshold = cfg_buf[0];
    int32_t want_max  = cfg_buf[1]; 

    constexpr int vector_width = 16;
    constexpr int total_elements = 1024;
    
    aie::vector<int32_t, vector_width> t_vec = aie::broadcast<int32_t, vector_width>(threshold);

    int valid_count = 0;
    int out_idx = 1;

    for (int i = 0; i < total_elements; i += vector_width) {
        aie::vector<int32_t, vector_width> v = aie::load_v<vector_width>(in_buf + i);
        
        aie::mask<vector_width> cmp_mask = want_max ? aie::gt(v, t_vec) : aie::lt(v, t_vec);
        
        int32_t tmp[vector_width];
        aie::store_v(tmp, v);
        
        #pragma unroll
        for (int j = 0; j < vector_width; ++j) {
            if (cmp_mask.test(j)) {
                out_buf[out_idx++] = tmp[j];
                valid_count++;
            }
        }
    }
    
    out_buf[0] = valid_count;
}

} // extern "C"
