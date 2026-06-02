#include <aie_api/aie.hpp>
#include <cstdint>

alignas(32) static const int32_t v_offsets[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int J>
inline aie::mask<16> get_blend_mask(int global_base_idx, int k) {
    aie::vector<int32_t, 16> idx = aie::add(aie::broadcast<int32_t, 16>(global_base_idx), aie::load_v<16>(v_offsets));
    aie::vector<int32_t, 16> asc_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(k));
    aie::mask<16> ascending = aie::eq(asc_and, aie::broadcast<int32_t, 16>(0));
    aie::vector<int32_t, 16> odd_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(J));
    aie::mask<16> odd = aie::neq(odd_and, aie::broadcast<int32_t, 16>(0));
    
    aie::vector<int32_t, 16> asc_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), ascending);
    aie::vector<int32_t, 16> odd_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), odd);
    
    return aie::eq(asc_int, odd_int);
}

template <int J>
inline aie::vector<int32_t, 16> permutex(const aie::vector<int32_t, 16>& v) {
    alignas(32) int32_t tmp[16];
    aie::store_v(tmp, v);
    alignas(32) int32_t out_tmp[16];
    #pragma unroll(16)
    for (int idx = 0; idx < 16; ++idx) out_tmp[idx] = tmp[idx ^ J];
    return aie::load_v<16>(out_tmp);
}

template <int J>
void process_simd(int32_t* restrict in_buf, int32_t* restrict out_buf, int32_t k, int32_t type, int32_t chunk_idx, int32_t pad_val) {
    constexpr int vector_width = 16;
    constexpr int total_elements = 1024;
    int base_global_idx = chunk_idx * total_elements;

    if (type == 1) { 
        const int32_t j_minus_1 = J - 1;
        const int32_t j_mask = ~j_minus_1;
        for (int i = 0; i < total_elements; i += 16) aie::store_v(out_buf + i, aie::broadcast<int32_t, 16>(pad_val));
        for (int i = 0; i < total_elements; ++i) {
            if ((i & J) != 0) continue; 
            int32_t ixj = i + J;
            if (ixj >= total_elements) continue;
            int32_t out_idx = ((i >> 1) & j_mask) | (i & j_minus_1);
            int32_t a = in_buf[i];
            int32_t b = in_buf[ixj];
            out_buf[out_idx] = (a < b) ? a : b; 
        }
        return;
    } 

    for (int i = 0; i < total_elements; i += vector_width) {
        aie::vector<int32_t, vector_width> v = aie::load_v<vector_width>(in_buf + i);
        aie::vector<int32_t, vector_width> swapped = permutex<J>(v);
        aie::vector<int32_t, vector_width> lo = aie::min(v, swapped);
        aie::vector<int32_t, vector_width> hi = aie::max(v, swapped);
        aie::mask<vector_width> mask = get_blend_mask<J>(base_global_idx + i, k);
        aie::vector<int32_t, vector_width> out = aie::select(lo, hi, mask);
        aie::store_v(out_buf + i, out);
    }
}

void process_scalar(int32_t* restrict in_buf, int32_t* restrict out_buf, int32_t j, int32_t k, int32_t type, int32_t chunk_idx, int32_t pad_val) {
    constexpr int total_elements = 1024;
    int base_global_idx = chunk_idx * total_elements;
    
    if (type == 1) { 
        const int32_t j_minus_1 = j - 1;
        const int32_t j_mask = ~j_minus_1;
        for (int i = 0; i < total_elements; ++i) out_buf[i] = pad_val;
        for (int i = 0; i < total_elements; ++i) {
            int global_i = base_global_idx + i;
            if ((global_i & j) != 0) continue;
            int32_t ixj = i + j;
            if (ixj >= total_elements) continue;
            int32_t out_idx = ((i >> 1) & j_mask) | (i & j_minus_1);
            int32_t a = in_buf[i];
            int32_t b = in_buf[ixj];
            out_buf[out_idx] = (a < b) ? a : b;
        }
        return;
    }

    for (int i = 0; i < total_elements; ++i) out_buf[i] = in_buf[i];
    for (int i = 0; i < total_elements; ++i) {
        int global_i = base_global_idx + i;
        if ((global_i & j) != 0) continue; 
        int32_t ixj = i + j;
        if (ixj >= total_elements) break;
        bool ascending = (global_i & k) == 0;
        if (ascending) {
            if (out_buf[i] > out_buf[ixj]) std::swap(out_buf[i], out_buf[ixj]);
        } else {
            if (out_buf[i] < out_buf[ixj]) std::swap(out_buf[i], out_buf[ixj]);
        }
    }
}

extern "C" {
void bitonic_step_kernel(int32_t* restrict in_buf, int32_t* restrict out_buf, int32_t* restrict cfg_buf) {
    int32_t j = cfg_buf[0];
    int32_t k = cfg_buf[1];
    int32_t type = cfg_buf[2]; 
    int32_t chunk_idx = cfg_buf[3];
    int32_t pad_val = cfg_buf[4];

    if (type == 2) { 
        int32_t global_base = chunk_idx; 
        for (int i = 0; i < 512; i += 16) {
            aie::vector<int32_t, 16> v1 = aie::load_v<16>(in_buf + i);
            aie::vector<int32_t, 16> v2 = aie::load_v<16>(in_buf + i + 512);
            aie::vector<int32_t, 16> lo = aie::min(v1, v2);
            aie::vector<int32_t, 16> hi = aie::max(v1, v2);
            aie::vector<int32_t, 16> idx = aie::add(aie::broadcast<int32_t, 16>(global_base + i), aie::load_v<16>(v_offsets));
            aie::vector<int32_t, 16> asc_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(k));
            aie::mask<16> mask = aie::neq(asc_and, aie::broadcast<int32_t, 16>(0));
            aie::vector<int32_t, 16> out1 = aie::select(lo, hi, mask);
            aie::vector<int32_t, 16> out2 = aie::select(hi, lo, mask);
            aie::store_v(out_buf + i, out1);
            aie::store_v(out_buf + i + 512, out2);
        }
        return;
    }
    
    if (type == 3) {
        for (int i = 0; i < 512; i += 16) {
            aie::vector<int32_t, 16> v1 = aie::load_v<16>(in_buf + i);
            aie::vector<int32_t, 16> v2 = aie::load_v<16>(in_buf + i + 512);
            aie::vector<int32_t, 16> winner = aie::min(v1, v2); 
            aie::store_v(out_buf + i, winner);
            aie::store_v(out_buf + i + 512, aie::broadcast<int32_t, 16>(pad_val));
        }
        return;
    }

    switch (j) {
        case 1:  process_simd<1>(in_buf, out_buf, k, type, chunk_idx, pad_val); break;
        case 2:  process_simd<2>(in_buf, out_buf, k, type, chunk_idx, pad_val); break;
        case 4:  process_simd<4>(in_buf, out_buf, k, type, chunk_idx, pad_val); break;
        case 8:  process_simd<8>(in_buf, out_buf, k, type, chunk_idx, pad_val); break;
        default: process_scalar(in_buf, out_buf, j, k, type, chunk_idx, pad_val); break;
    }
}
}
