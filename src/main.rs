#![allow(unused)]
// use rayon::prelude::*;
// use rayon::slice::ParallelSlice;
// use stellar::algebra::ndmethods::{basic_mult, create_identity_matrix, tensor_mult};
// use stellar::algebra::ndmethods::{lt_matrix_mult, matrix_mult};
// use stellar::decomposition::lq::AutumnDecomp;
// use stellar::equality::approximate::approx_vector_eq;
// use stellar::random::generation::generate_random_matrix;
// use stellar::structure::ndarray::NdArray;
// use std::simd;

// TODO: make the little kernel method
// then make the LX, async method

/// kernel_mult
/// a * b -> c
///
/// * a : block of a
/// * b : block of b
/// * c : block row of c
/// * block : size of block rows which is equal to block cols
/// * stride : the number of cols in the output matrix c
/// * offset : the outer k which will determine where we need to write
use stellar::random::generation::generate_random_vector;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
const BLOCK_KERNEL: usize = 8;

pub fn kernel_mult(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_i: usize,
    block_k: usize,
    block_j: usize,
    stride: usize,
    offset: usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let block = block_i;
        kernel_mult_avx(a, b, c, block, stride, offset);
    }

    #[cfg(not(target_arch = "x86_64"))]
    kernel_mult_scalar(a, b, c, block_i, block_k, block_j, stride, offset);
}

#[inline(always)]
unsafe fn kernel_mult_avx(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block: usize,
    stride: usize,
    offset: usize,
) {
    unsafe {
        let mut aoffset = 0;
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        let cptr = c.as_mut_ptr().add(offset);
        let i_row = _mm256_loadu_ps(bptr);
        let ii_row = _mm256_loadu_ps(bptr.add(8));
        let iii_row = _mm256_loadu_ps(bptr.add(16));
        let iv_row = _mm256_loadu_ps(bptr.add(24));
        let v_row = _mm256_loadu_ps(bptr.add(36));
        let vi_row = _mm256_loadu_ps(bptr.add(42));
        let vii_row = _mm256_loadu_ps(bptr.add(48));
        let viii_row = _mm256_loadu_ps(bptr.add(48));

        let mut aoffset = 0;
        let mut coffset = offset;
        for i in 0..block {
            let scalar = 13f32;
            let arow = aptr.add(aoffset);
            let c_row = cptr.add(coffset);
            let mut acc = _mm256_loadu_ps(c_row);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(0)), i_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(1)), ii_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(2)), iii_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(3)), iv_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(4)), v_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(5)), vi_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(6)), vii_row, acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(7)), viii_row, acc);
            _mm256_store_ps(c_row, acc);
            aoffset += 8;
            coffset += stride;
        }
    }
}
#[inline(always)]
pub fn kernel_mult_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_i: usize,
    block_k: usize,
    block_j: usize,
    stride: usize,
    offset: usize,
) {
    // simple method do benchmark to see if quicker
    // c must be filled with zeros prior to overwriting just using this to benchmark
    let mut aoffset = 0;
    let mut coffset = offset;
    let mut boffset;
    for _i in 0..block_i {
        boffset = 0;
        let a_row = &a[aoffset..aoffset + block_k];
        for k in 0..block_k {
            let scalar = a_row[k];
            let b_row = &b[boffset..boffset + block_k];
            let c_row = &mut c[coffset..coffset + block_j];
            for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
                *c += scalar * b;
            }
            boffset += block_j;
        }
        aoffset += block_k;
        coffset += stride;
    }
}

fn test_performance() {
    let block = 8;
    let stride = 8;
    let offset = 0;
    let a = generate_random_vector(8);
    let b = generate_random_vector(8);
    let mut c_avx = generate_random_vector(8);
    let mut c_scaler = generate_random_vector(8);
    unsafe {
        kernel_mult_avx(&a, &b, &mut c_avx, block, stride, offset);
    }
    kernel_mult_scalar(&a, &b, &mut c_scaler, block, block, block, stride, offset);
}

fn main() {}
