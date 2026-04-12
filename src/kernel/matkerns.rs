use crate::arch::SIMD_WIDTH;
use crate::kernel::avx2::kernel_mult_avx2;
use crate::kernel::avx512::kernel_mult_avx512;

#[cfg(target_arch = "x86_64")]

pub fn kernel_mult_in_progress(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    #[cfg(feature = "avx512")]
    unsafe {
        if 16 == block_n && 16 == block_k {
            return kernel_mult_avx512(a, b, c, block_m, s_x, s_y);
        }
    }
    #[cfg(feature = "avx2")]
    unsafe {
        if 8 == block_n && 8 == block_k {
            return kernel_mult_avx2(a, b, c, block_m, s_x, s_y);
        }
    }
    kernel_mult_scalar(a, b, c, block_m, block_k, block_n, s_x, s_y);
}

pub fn kernel_mult(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    #[cfg(feature = "avx512")]
    unsafe {
        if 16 == block_n && 16 == block_k {
            // if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            return kernel_mult_avx512(a, b, c, block_m, s_x, s_y);
            // }
        }
    }
    #[cfg(feature = "avx2")]
    unsafe {
        if 8 == block_n && 8 == block_k {
            return kernel_mult_avx2(a, b, c, block_m, s_x, s_y);
        }
    }
    kernel_mult_scalar(a, b, c, block_m, block_k, block_n, s_x, s_y);
}

// pub fn kernel_mult(
//     a: &[f32],
//     b: &[f32],
//     c: &mut [f32],
//     block_m: usize,
//     block_k: usize,
//     block_n: usize,
//     stride: usize,
//     offset: usize,
// ) {
//     #[cfg(target_arch = "x86_64")]
//     unsafe {
//         if BLOCK_AVX2 == block_n && block_n == block_k {
//             // if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
//             return kernel_mult_avx2(a, b, c, block_m, stride, offset);
//             // }
//         }
//     }
//     #[cfg(target_arch = "x86_64")]
//     // TODO: change input of b to not be block and update the kernel mult avx
//     // if it is edge case then copy into workspace_y here, and pass that in as b;
//     // just make a fn which does this so this doesn't explode in defn

//     kernel_mult_scalar(a, b, c, block_m, block_k, block_n, stride, offset);
// }

/// kernel_mult
/// a * b -> c
///
/// * a : block of a
/// * b : block of b
/// * c : block row of c
/// * block : size of block rows which is equal to block cols
/// * stride : the number of cols in the output matrix c
/// * offset : the outer k which will determine where we need to write
#[inline(always)]
pub fn kernel_mult_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    // simple method to handle edge cases
    let mut aoffset = 0;
    let mut coffset = 0;
    let mut boffset;
    for _i in 0..block_m {
        boffset = 0;
        let a_row = &a[aoffset..aoffset + s_x];
        for k in 0..block_k {
            let scalar = a_row[k];
            // let b_row = &b[boffset..boffset + block_n];
            let b_row = &b[boffset..boffset + s_y];
            let c_row = &mut c[coffset..coffset + block_n];
            for (c, b) in c_row.iter_mut().zip(b_row.iter()) {
                *c += scalar * b;
            }
            // boffset += BLOCK_AVX2;
            boffset += s_y;
        }
        aoffset += s_x;
        coffset += s_y;
    }
}
