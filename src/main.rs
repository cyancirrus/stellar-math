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

use std::ptr::copy_nonoverlapping;
use stellar::algebra::bmethods::pack;
use stellar::arch::SIMD_WIDTH;
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use stellar::kernel::avx2::constants::{MASK, cfma_accum, fma_accum, mask_load, mask_store_ctrl};
use stellar::random::generation::generate_random_matrix;
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use stellar_macros::{kernel_lmult_lower_tri, kernel_mult_alligned, kernel_mult_unalligned};
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// const LC: usize = 48;
// const MC: usize = 48;
// const PC: usize = 32;
// const NC: usize = 96;
const MC: usize = 48;
const PC: usize = 256;
use std::arch::x86_64::{
    __m256, __m256i, _mm256_and_ps, _mm256_blendv_ps, _mm256_broadcast_ss, _mm256_castsi256_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_ps, _mm256_maskstore_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_stream_ps,
};
#[inline(always)]
#[cfg(not(any(feature = "avx2")))]
fn default_mult(bptr: *mut f32, dptr: *const f32, re: usize, se: usize, s_b: usize, s_d: usize) {
    unsafe {
        println!("recompile with avx2");
    }
}

pub fn kernel_umult(
    mut xptr: *const f32,
    mut yptr: *mut f32,
    tptr: *mut f32,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
    kernel_mult_unalligned!(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
    #[cfg(not(any(feature = "avx2")))]
    println!("hey run it with flag for arch");
    // default_mult(bptr, dptr, re, se, PC, s_d);
}
// pub fn kernel_mult(
//     mut xptr: *const f32,
//     mut yptr: *mut f32,
//     tptr: *mut f32,
//     m: usize,
//     p: usize,
//     n: usize,
//     s_x: usize,
//     s_y: usize,
//     s_t: usize,
// ) {
//     #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
//     kernel_mult_alligned!(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
//     #[cfg(not(any(feature = "avx2")))]
//     println!("hey run it with flag for arch");
//     // default_mult(bptr, dptr, re, se, PC, s_d);
// }
// pub fn kernel_lmult_lt(
//     mut xptr: *const f32,
//     mut yptr: *mut f32,
//     tptr: *mut f32,
//     m: usize,
//     p: usize,
//     n: usize,
//     s_x: usize,
//     s_y: usize,
//     s_t: usize,
// ) {
//     #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
//     kernel_lmult_lower_tri!(xptr, yptr, tptr, m, p, n, s_x, s_y, s_t);
//     #[cfg(not(any(feature = "avx2")))]
//     println!("hey run it with flag for arch");
//     // default_mult(bptr, dptr, re, se, PC, s_d);
// }

fn main() {}
