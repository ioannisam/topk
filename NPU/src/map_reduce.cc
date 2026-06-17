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

    for (int i = 0; i < total_elements; i += vector_width) {
        aie::vector<int32_t, vector_width> v = aie::load_v<vector_width>(in_buf + i);
        aie::mask<vector_width> cmp_mask = want_max ? aie::gt(v, t_vec) : aie::lt(v, t_vec);
        aie::vector<int32_t, vector_width> out_v = aie::select(sentinel_vec, v, cmp_mask);
        aie::store_v(out_buf + i, out_v);
    }
}

} // extern "C"
