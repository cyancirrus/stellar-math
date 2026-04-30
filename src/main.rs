#![allow(unused)]
// TODO:
// then make the LX, async method
// do the 16 x 16 instruction ie 512 for the tower
// make the toml cfg to get cacheline size etc
// do a small test
// inspect the flamegraph to see if any hanging threads
// ie suspect like communication jam in l1-> l2
//
// value sanity start working on the LX async vision with the queue

// 1. Animate demo        ← most legible to employers
// 2. Blog redesign       ← makes everything else findable
// 3. Triangle kernel     ← 2hrs, unblocks LQ block
// 4. Trait refactor      ← important but least urgent
use stellar::arch::SIMD_WIDTH;

use stellar::algebra::ndmethods::basic_mult;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
// #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_and_ps, _mm256_broadcast_ss,
    _mm256_castsi256_ps, _mm256_fmadd_ps, _mm256_loadu_si256, _mm256_maskload_ps,
    _mm256_maskstore_ps, _mm256_set1_epi32, _mm256_setzero_ps,
};

// negative 1 is twos complement so all bits active
#[rustfmt::skip]
const MASK:[[i32;8];9] = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0,  0,  0],
    [-1, -1,  0,  0,  0,  0,  0,  0],
    [-1, -1, -1,  0,  0,  0,  0,  0],
    [-1, -1, -1, -1,  0,  0,  0,  0],
    [-1, -1, -1, -1, -1,  0,  0,  0],
    [-1, -1, -1, -1, -1, -1,  0,  0],
    [-1, -1, -1, -1, -1, -1, -1,  0],
    [-1, -1, -1, -1, -1, -1, -1, -1],
];

static ZEROS: [f32; 8] = [0f32; 8];
unsafe fn gate_row(ptr: *const f32, ctrl: i32, mask: __m256i) -> __m256 {
    unsafe {
        let safe_ptr = if ctrl != 0 { ptr } else { ZEROS.as_ptr() };
        _mm256_maskload_ps(safe_ptr, mask)
    }
}
unsafe fn sgate_row(ptr: *mut f32, ctrl: i32, mask: __m256i, data: __m256) {
    unsafe {
        if ctrl != 0 {
            _mm256_maskstore_ps(ptr, mask, data);
        }
    }
}
macro_rules! fma_gated {
    ($acc:expr, $ptr:expr, $mask_bit:expr, $data:expr) => {
        if $mask_bit != 0 {
            $acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&*$ptr), $data, $acc);
            // $acc = _mm256_fmadd_ps(_mm256_set1_ps(*$ptr), $b, $acc);
        }
    };
}
#[target_feature(enable = "avx,avx2,fma")]
pub fn kernel_imult_lt_safe(
    mut xptr: *const f32,
    mut yptr: *const f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // Sum[K] Union[I] { g^i = aik b^k }
    // excels at processing panels of data ie 8 x K * K x 8;
    unsafe {
        let mask_n_ptr = MASK[n].as_ptr() as *const __m256i;
        let mask_n = _mm256_loadu_si256(mask_n_ptr);
        let mut i_row = _mm256_maskload_ps(tptr, mask_n);
        let mut v_row = _mm256_maskload_ps(tptr.add(s_t * 4), mask_n);
        let mut ii_row = _mm256_maskload_ps(tptr.add(s_t), mask_n);
        let mut vi_row = _mm256_maskload_ps(tptr.add(s_t * 5), mask_n);
        let mut iii_row = _mm256_maskload_ps(tptr.add(s_t * 2), mask_n);
        let mut vii_row = _mm256_maskload_ps(tptr.add(s_t * 6), mask_n);
        let mut iv_row = _mm256_maskload_ps(tptr.add(s_t * 3), mask_n);
        let mut viii_row = _mm256_maskload_ps(tptr.add(s_t * 7), mask_n);
        let mask_m = MASK[m];
        for k in 8..p {
            // _mm_prefetch(yptr.add(s_y) as *const i8, _MM_HINT_T0);
            // _mm_prefetch(xptr.add(4 * s_x) as *const i8, _MM_HINT_T0);
            let b0 = _mm256_maskload_ps(yptr, mask_n);
            yptr = yptr.add(s_y );
            fma_gated!(i_row, xptr, mask_m[0],b0);
            fma_gated!(ii_row, xptr.add(s_x), mask_m[1],b0);
            fma_gated!(iii_row, xptr.add(2 * s_x), mask_m[2],b0);
            fma_gated!(iv_row, xptr.add(3 * s_x), mask_m[3],b0);
            fma_gated!(v_row, xptr.add(4 * s_x), mask_m[4],b0);
            fma_gated!(vi_row, xptr.add(5 * s_x), mask_m[5],b0);
            fma_gated!(vii_row, xptr.add(6 * s_x), mask_m[6],b0);
            fma_gated!(viii_row, xptr.add(7 * s_x), mask_m[7],b0);
            xptr = xptr.add(1);
        }
        let mut mask_t = mask_m;
        let mut idx = 0;
        for k in 0..8.min(p) {
            let b0 = _mm256_maskload_ps(yptr, mask_n);
            yptr = yptr.add(s_y );
            fma_gated!(i_row, xptr, mask_t[0], b0);
            fma_gated!(ii_row, xptr.add(s_x), mask_t[1], b0);
            fma_gated!(iii_row, xptr.add(2 * s_x), mask_t[2], b0);
            fma_gated!(iv_row, xptr.add(3 * s_x), mask_t[3], b0);
            fma_gated!(v_row, xptr.add(4 * s_x), mask_t[4], b0);
            fma_gated!(vi_row, xptr.add(5 * s_x), mask_t[5], b0);
            fma_gated!(vii_row, xptr.add(6 * s_x), mask_t[6], b0);
            fma_gated!(viii_row, xptr.add(7 * s_x), mask_t[7], b0);
            mask_t[k] = 0;
            xptr = xptr.add(1);
        }
        sgate_row(tptr, mask_m[0], mask_n, i_row);
        sgate_row(tptr.add(s_t * 4), mask_m[4], mask_n, v_row);
        sgate_row(tptr.add(s_t), mask_m[1], mask_n, ii_row);
        sgate_row(tptr.add(s_t * 5), mask_m[5], mask_n, vi_row);
        sgate_row(tptr.add(s_t * 2), mask_m[2], mask_n, iii_row);
        sgate_row(tptr.add(s_t * 6), mask_m[6], mask_n, vii_row);
        sgate_row(tptr.add(s_t * 3), mask_m[3], mask_n, iv_row);
        sgate_row(tptr.add(s_t * 7), mask_m[7], mask_n, viii_row);
    }
}

fn main() {}
