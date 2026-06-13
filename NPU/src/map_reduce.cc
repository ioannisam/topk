#include <aie_api/aie.hpp>
#include <cstdint>

extern "C" {

void map_reduce_step_kernel(int32_t* restrict in_buf, int32_t* restrict out_buf, int32_t* restrict cfg_buf) {
    int32_t threshold = cfg_buf[0];
    int32_t want_max  = cfg_buf[1]; 

    constexpr int vector_width = 16;
    constexpr int total_elements = 1024;
    
    aie::vector<int32_t, vector_width> t_vec = aie::broadcast<int32_t, vector_width>(threshold);

    int32_t valid_count = 0;
    int32_t* out_data_ptr = out_buf + 1; 

    for (int i = 0; i < total_elements; i += vector_width) {
        aie::vector<int32_t, vector_width> v = aie::load_v<vector_width>(in_buf + i);
        aie::mask<vector_width> cmp_mask = want_max ? aie::gt(v, t_vec) : aie::lt(v, t_vec);
        
        for (int lane = 0; lane < vector_width; ++lane) {
            if (cmp_mask.test(lane)) {
                *out_data_ptr++ = v[lane];
                valid_count++;
            }
        }
    }
    
    // Write the total number of vald elements at the very beginning of the chunk
    out_buf[0] = valid_count;
}

} // extern "C"
