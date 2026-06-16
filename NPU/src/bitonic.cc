#include <aie_api/aie.hpp>
#include <cstdint>

alignas(32) static const int32_t v_offsets[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int J>
inline aie::vector<int32_t, 16> shuffle_v(const aie::vector<int32_t, 16>& v) {
    alignas(32) int32_t tmp[16], out_tmp[16];
    aie::store_v(tmp, v);
    #pragma unroll(16)
    for (int idx = 0; idx < 16; ++idx) out_tmp[idx] = tmp[idx ^ J];
    return aie::load_v<16>(out_tmp);
}

template <int J, int K>
inline aie::mask<16> get_blend_mask(int global_base_idx) {
    aie::vector<int32_t, 16> idx = aie::add(aie::broadcast<int32_t, 16>(global_base_idx), aie::load_v<16>(v_offsets));
    aie::vector<int32_t, 16> asc_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(K));
    aie::mask<16> ascending = aie::eq(asc_and, aie::broadcast<int32_t, 16>(0));
    aie::vector<int32_t, 16> odd_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(J));
    aie::mask<16> odd = aie::neq(odd_and, aie::broadcast<int32_t, 16>(0));
    aie::vector<int32_t, 16> asc_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), ascending);
    aie::vector<int32_t, 16> odd_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), odd);
    return aie::eq(asc_int, odd_int);
}

template <int J, int K>
inline void bitonic_step_intra(int32_t* __restrict data) {
    for (int i = 0; i < 1024; i += 16) {
        aie::vector<int32_t, 16> v = aie::load_v<16>(data + i);
        aie::vector<int32_t, 16> swapped = shuffle_v<J>(v);
        aie::vector<int32_t, 16> lo = aie::min(v, swapped);
        aie::vector<int32_t, 16> hi = aie::max(v, swapped);
        aie::vector<int32_t, 16> result = aie::select(lo, hi, get_blend_mask<J, K>(i));
        aie::store_v(data + i, result);
    }
}

template <int J, int K>
inline void bitonic_step_inter(int32_t* __restrict data) {
    for (int i = 0; i < 1024; i += (2 * J)) {
        bool ascending = ((i & K) == 0);
        for (int offset = 0; offset < J; offset += 16) {
            int idx1 = i + offset;
            int idx2 = i + J + offset;
            aie::vector<int32_t, 16> v1 = aie::load_v<16>(data + idx1);
            aie::vector<int32_t, 16> v2 = aie::load_v<16>(data + idx2);
            aie::vector<int32_t, 16> lo = aie::min(v1, v2);
            aie::vector<int32_t, 16> hi = aie::max(v1, v2);
            if (ascending) {
                aie::store_v(data + idx1, lo); aie::store_v(data + idx2, hi);
            } else {
                aie::store_v(data + idx1, hi); aie::store_v(data + idx2, lo);
            }
        }
    }
}

extern "C" {

void pipeline_core_1(int32_t* __restrict in, int32_t* __restrict out) {
    for (int i = 0; i < 1024; i += 16) aie::store_v(out + i, aie::load_v<16>(in + i));

    bitonic_step_intra<1, 2>(out);
    bitonic_step_intra<2, 4>(out); bitonic_step_intra<1, 4>(out);
    bitonic_step_intra<4, 8>(out); bitonic_step_intra<2, 8>(out); bitonic_step_intra<1, 8>(out);
    bitonic_step_intra<8, 16>(out); bitonic_step_intra<4, 16>(out); bitonic_step_intra<2, 16>(out); bitonic_step_intra<1, 16>(out);
    bitonic_step_inter<16, 32>(out); bitonic_step_intra<8, 32>(out); bitonic_step_intra<4, 32>(out); bitonic_step_intra<2, 32>(out); bitonic_step_intra<1, 32>(out);
}

void pipeline_core_2(int32_t* __restrict in, int32_t* __restrict out) {
    for (int i = 0; i < 1024; i += 16) aie::store_v(out + i, aie::load_v<16>(in + i));

    bitonic_step_inter<32, 64>(out); bitonic_step_inter<16, 64>(out); bitonic_step_intra<8, 64>(out); bitonic_step_intra<4, 64>(out); bitonic_step_intra<2, 64>(out); bitonic_step_intra<1, 64>(out);
    bitonic_step_inter<64, 128>(out); bitonic_step_inter<32, 128>(out); bitonic_step_inter<16, 128>(out); bitonic_step_intra<8, 128>(out); bitonic_step_intra<4, 128>(out); bitonic_step_intra<2, 128>(out); bitonic_step_intra<1, 128>(out);
}

void pipeline_core_3(int32_t* __restrict in, int32_t* __restrict out) {
    for (int i = 0; i < 1024; i += 16) aie::store_v(out + i, aie::load_v<16>(in + i));

    bitonic_step_inter<128, 256>(out); bitonic_step_inter<64, 256>(out); bitonic_step_inter<32, 256>(out); bitonic_step_inter<16, 256>(out); bitonic_step_intra<8, 256>(out); bitonic_step_intra<4, 256>(out); bitonic_step_intra<2, 256>(out); bitonic_step_intra<1, 256>(out);
    bitonic_step_inter<256, 512>(out); bitonic_step_inter<128, 512>(out); bitonic_step_inter<64, 512>(out); bitonic_step_inter<32, 512>(out); bitonic_step_inter<16, 512>(out); bitonic_step_intra<8, 512>(out); bitonic_step_intra<4, 512>(out); bitonic_step_intra<2, 512>(out); bitonic_step_intra<1, 512>(out);
}

void pipeline_core_4(int32_t* __restrict in, int32_t* __restrict out) {
    for (int i = 0; i < 1024; i += 16) aie::store_v(out + i, aie::load_v<16>(in + i));

    bitonic_step_inter<512, 1024>(out); bitonic_step_inter<256, 1024>(out); bitonic_step_inter<128, 1024>(out); bitonic_step_inter<64, 1024>(out); bitonic_step_inter<32, 1024>(out); bitonic_step_inter<16, 1024>(out); bitonic_step_intra<8, 1024>(out); bitonic_step_intra<4, 1024>(out); bitonic_step_intra<2, 1024>(out); bitonic_step_intra<1, 1024>(out);
}

}
