#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_storeu_ps,
};
const BLOCK_KERNEL: usize = 8;

pub fn kernel_mult(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    stride: usize,
    offset: usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if BLOCK_KERNEL == block_n && block_n == block_k {
            // if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            return kernel_mult_avx(a, b, c, block_m, stride, offset);
            // }
        }
    }
    // TODO: change input of b to not be block and update the kernel mult avx
    // if it is edge case then copy into workspace_y here, and pass that in as b;
    // just make a fn which does this so this doesn't explode in defn

    kernel_mult_scalar(a, b, c, block_m, block_k, block_n, stride, offset);
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
    unsafe {
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        let cptr = c.as_mut_ptr();
        // let i_row = _mm256_loadu_ps(bptr);
        let i_row = _mm256_loadu_ps(bptr);
        let ii_row = _mm256_loadu_ps(bptr.add(stride));
        let iii_row = _mm256_loadu_ps(bptr.add(stride * 2));
        let iv_row = _mm256_loadu_ps(bptr.add(stride * 3));
        let v_row = _mm256_loadu_ps(bptr.add(stride * 4));
        let vi_row = _mm256_loadu_ps(bptr.add(stride * 5));
        let vii_row = _mm256_loadu_ps(bptr.add(stride * 6));
        let viii_row = _mm256_loadu_ps(bptr.add(stride * 7));

        let mut aoffset = 0;
        let mut coffset = offset;
        for _ in 0..block_v {
            let arow = aptr.add(aoffset);
            let c_row = cptr.add(coffset);
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_loadu_ps(c_row);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(0)), i_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(1)), ii_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(2)), iii_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(3)), iv_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(4)), v_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(5)), vi_row, acc1);
            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(6)), vii_row, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*arow.add(7)), viii_row, acc1);
            _mm256_storeu_ps(c_row, _mm256_add_ps(acc0, acc1));
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
    block_m: usize,
    block_k: usize,
    block_n: usize,
    stride: usize,
    offset: usize,
) {
    // simple method to handle edge cases
    let mut aoffset = 0;
    let mut coffset = offset;
    let mut boffset;
    for _i in 0..block_m {
        boffset = 0;
        let a_row = &a[aoffset..aoffset + BLOCK_KERNEL];
        for k in 0..block_k {
            let scalar = a_row[k];
            // let b_row = &b[boffset..boffset + block_n];
            let b_row = &b[boffset..boffset + stride];
            let c_row = &mut c[coffset..coffset + block_n];
            for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
                *c += scalar * b;
            }
            // boffset += BLOCK_KERNEL;
            boffset += stride;
        }
        aoffset += BLOCK_KERNEL;
        coffset += stride;
    }
}
