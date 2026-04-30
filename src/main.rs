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

fn main() {}
