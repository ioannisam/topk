#include <aie_api/aie.hpp>
#include <cstdint>

constexpr int BLOCK = 16;

alignas(32) static const int32_t v_offsets[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

template <int J> inline aie::mask<16> partner_select_mask() {
	aie::vector<int32_t, 16> offs = aie::load_v<16>(v_offsets);
	return aie::neq(aie::bit_and(offs, aie::broadcast<int32_t, 16>(J)), aie::broadcast<int32_t, 16>(0));
}

template <int J, int K> inline aie::mask<16> get_blend_mask() {
	aie::vector<int32_t, 16> idx = aie::bit_and(aie::load_v<16>(v_offsets), aie::broadcast<int32_t, 16>(BLOCK - 1));
	aie::vector<int32_t, 16> asc_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(K));
	aie::mask<16> ascending = aie::eq(asc_and, aie::broadcast<int32_t, 16>(0));
	aie::vector<int32_t, 16> odd_and = aie::bit_and(idx, aie::broadcast<int32_t, 16>(J));
	aie::mask<16> odd = aie::neq(odd_and, aie::broadcast<int32_t, 16>(0));
	aie::vector<int32_t, 16> asc_int =
		aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), ascending);
	aie::vector<int32_t, 16> odd_int = aie::select(aie::broadcast<int32_t, 16>(0), aie::broadcast<int32_t, 16>(1), odd);
	return aie::eq(asc_int, odd_int);
}

template <int J, int K> inline aie::vector<int32_t, 16> compare_exchange(const aie::vector<int32_t, 16>& v) {
	aie::vector<int32_t, 16> partner =
		aie::select(aie::shuffle_down(v, J), aie::shuffle_up(v, J), partner_select_mask<J>());
	aie::vector<int32_t, 16> lo = aie::min(v, partner);
	aie::vector<int32_t, 16> hi = aie::max(v, partner);
	return aie::select(lo, hi, get_blend_mask<J, K>());
}

inline aie::vector<int32_t, 16> sort_block(aie::vector<int32_t, 16> v) {
	v = compare_exchange<1, 2>(v);
	v = compare_exchange<2, 4>(v);
	v = compare_exchange<1, 4>(v);
	v = compare_exchange<4, 8>(v);
	v = compare_exchange<2, 8>(v);
	v = compare_exchange<1, 8>(v);
	v = compare_exchange<8, 16>(v);
	v = compare_exchange<4, 16>(v);
	v = compare_exchange<2, 16>(v);
	v = compare_exchange<1, 16>(v);
	return v;
}

extern "C" {

void bitonic_sort_runs(int32_t* __restrict in, int32_t* __restrict out) {
#pragma unroll(4)
	for (int i = 0; i < 1024; i += 16) {
		aie::store_v(out + i, sort_block(aie::load_v<16>(in + i)));
	}
}
}
