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
use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_castpd_ps, _mm256_castps_pd,
    _mm256_castsi256_ps, _mm256_fmadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_mask_load_ps, _mm256_mask_loadu_ps, _mm256_maskload_ps, _mm256_permute2f128_ps,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_unpackhi_pd, _mm256_unpackhi_ps,
    _mm256_unpacklo_pd, _mm256_unpacklo_ps,
};
use stellar::algebra::bmethods::tensor_blockkern;
use stellar::arch::SIMD_WIDTH;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
use stellar::algebra::ndmethods::basic_mult;
#[cfg(feature = "avx2")]
use stellar::kernel::{avx2, avx2safe};

fn zero_for_debug(a: &mut NdArray) {
    let a_d = &mut a.data;

    for i in 0..4 {
        for j in 4..8 {
            a_d[i * 8 + j] = 0f32;
        }
    }
    for i in 4..8 {
        for j in 0..8 {
            a_d[i * 8 + j] = 0f32;
        }
    }
}

#[cfg(feature = "avx2")]
fn test_mpn_kernels(m:usize, p:usize, n:usize) {
    unsafe {
    let (s_x, s_y, s_z) = (p,n,n);
    let mut x = generate_random_matrix(m, p);
    let mut y = generate_random_matrix(p, n);
    // zero_for_debug(&mut x);
    // zero_for_debug(&mut y);
    println!("x {x:?}");
    println!("y {y:?}");
    let mut x_simd = x.data.clone();
    let mut y_simd = y.data.clone();
    let mut w = vec![0f32; 8 * 8];
    let mut t = vec![0f32; m * n];
    // avx2safe::kernel_mult_safe(x_simd.as_ptr(), y_simd.as_ptr(), t.as_mut_ptr(), w.as_mut_ptr(), m, p, n, s_x, s_y, s_z);
    // avx2safe::kernel_mult_safe(x_simd.as_ptr(), y_simd.as_ptr(), t.as_mut_ptr(), w.as_mut_ptr(), m, p, n, s_x, s_y, s_z);
    // avx2::kernel_mult_simd(x_simd.as_ptr(), y_simd.as_ptr(), t.as_mut_ptr(), m, s_x, s_y, s_z);
    avx2::kernel_imult_simd(x_simd.as_ptr(), y_simd.as_ptr(), t.as_mut_ptr(), m, s_x, s_y, s_z);
    let expect = basic_mult(&x, &y);
    let inspect = NdArray { dims: vec![m,n], data: t.clone() };
    println!("expect {expect:?}");
    println!("actual {inspect:?}");
    assert!(approx_vector_eq(&expect.data, &t));
    }
}

#[cfg(feature = "avx2")]
fn test_kernels() {
    let dims = [
        (8, 8, 8),
        // (6, 4, 4),
        // (4, 6, 4),
        // (4, 4, 6),
        // (6, 4, 6),
        // (1, 8, 8),
        // (8, 1, 8),
        // (8, 8, 1),
        // (1, 8, 6),
        // (1, 1, 1),
    ];
    for (m, p, n) in dims {
        test_mpn_kernels(m, p, n);
    }

}

fn main() {
    #[cfg(feature = "avx2")]
    test_kernels();
}
