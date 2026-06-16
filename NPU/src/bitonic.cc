#include <aie_api/aie.hpp>
#include <cstdint>

constexpr int BLOCK = 16;

alignas(32) static const int32_t v_offsets[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int J>
inline aie::mask<16> partner_select_mask() {
    aie::vector<int32_t, 16> offs = aie::load_v<16>(v_offsets);
    return aie::neq(aie::bit_and(offs, aie::broadcast<int32_t, 16>(J)), aie::broadcast<int32_t, 16>(0));
}

template <int J>
inline aie::vector<int32_t, 16> butterfly_partner(const aie::vector<int32_t, 16>& v) {
    aie::vector<int32_t, 16> down = aie::shuffle_down(v, J);
    aie::vector<int32_t, 16> up = aie::shuffle_up(v, J);
    return aie::select(down, up, partner_select_mask<J>());
}

template <int J, int K>
inline aie::mask<16> get_blend_mask() {
    aie::vector<int32_t, 16> idx = aie::bit_and(aie::load_v<16>(v_offsets), aie::broadcast<int32_t, 16>(BLOCK - 1));
    aie::vector<int32_t, 16> asc_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(K));
    aie::mask<16> ascending = aie::eq(asc_and, aie::broadcast<int32_t, 16>(0));
    aie::vector<int32_t, 16> odd_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(J));
    aie::mask<16> odd = aie::neq(odd_and, aie::broadcast<int32_t, 16>(0));
    aie::vector<int32_t, 16> asc_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), ascending);
    aie::vector<int32_t, 16> odd_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), odd);
    return aie::eq(asc_int, odd_int);
}

template <int J, int K>
inline void bitonic_step(const int32_t* src, int32_t* dst) {
    aie::mask<16> blend = get_blend_mask<J, K>();
    for (int i = 0; i < 1024; i += 16) {
        aie::vector<int32_t, 16> v = aie::load_v<16>(src + i);
        aie::vector<int32_t, 16> swapped = butterfly_partner<J>(v);
        aie::vector<int32_t, 16> lo = aie::min(v, swapped);
        aie::vector<int32_t, 16> hi = aie::max(v, swapped);
        aie::store_v(dst + i, aie::select(lo, hi, blend));
    }
}

extern "C" {

void bitonic_sort_runs(int32_t* __restrict in, int32_t* __restrict out) {
    bitonic_step<1, 2>(in, out);
    bitonic_step<2, 4>(out, out); bitonic_step<1, 4>(out, out);
    bitonic_step<4, 8>(out, out); bitonic_step<2, 8>(out, out); bitonic_step<1, 8>(out, out);
    bitonic_step<8, 16>(out, out); bitonic_step<4, 16>(out, out); bitonic_step<2, 16>(out, out); bitonic_step<1, 16>(out, out);
}

}
