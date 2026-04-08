#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
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
        if BLOCK_KERNEL == block_j && block_j == block_k {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                return kernel_mult_avx(a, b, c, block_i, stride, offset);
            }
        }
    }

    kernel_mult_scalar(a, b, c, block_i, block_k, block_j, stride, offset);
}

/// kernel_mult
/// a * b -> c
///
/// * a : block of a
/// * b : block of b
/// * c : block row of c
/// * block : size of block rows which is equal to block cols
/// * stride : the number of cols in the output matrix c
/// * offset : the outer k which will determine where we need to write
#[target_feature(enable = "avx,fma")]
pub fn kernel_mult_avx(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_v: usize,
    stride: usize,
    offset: usize,
) {
    println!("in avx!");
    unsafe {
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        let cptr = c.as_mut_ptr();
        let i_row = _mm256_loadu_ps(bptr);
        let ii_row = _mm256_loadu_ps(bptr.add(8));
        let iii_row = _mm256_loadu_ps(bptr.add(16));
        let iv_row = _mm256_loadu_ps(bptr.add(24));
        let v_row = _mm256_loadu_ps(bptr.add(32));
        let vi_row = _mm256_loadu_ps(bptr.add(40));
        let vii_row = _mm256_loadu_ps(bptr.add(48));
        let viii_row = _mm256_loadu_ps(bptr.add(56));

        let mut aoffset = 0;
        let mut coffset = offset;
        for _ in 0..block_v {
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
            _mm256_storeu_ps(c_row, acc);
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
    // simple method to handle edge cases
    let mut aoffset = 0;
    let mut coffset = offset;
    let mut boffset;
    for _i in 0..block_i {
        boffset = 0;
        let a_row = &a[aoffset..aoffset + BLOCK_KERNEL];
        for k in 0..block_k {
            let scalar = a_row[k];
            let b_row = &b[boffset..boffset + block_j];
            let c_row = &mut c[coffset..coffset + block_j];
            for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
                *c += scalar * b;
            }
            boffset += BLOCK_KERNEL;
        }
        aoffset += BLOCK_KERNEL;
        coffset += stride;
    }
}
