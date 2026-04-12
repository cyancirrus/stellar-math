use crate::arch::SIMD_WIDTH;

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use crate::kernel::avx2::kernel_mult_simd;
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
use crate::kernel::avx512::kernel_mult_simd;
#[cfg(not(any(feature = "avx2", feature = "avx512", feature = "neon")))]
use crate::kernel::default::kernel_mult_simd;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
use crate::kernel::neon::kernel_mult_simd;

use crate::kernel::default::kernel_mult_scalar;

pub fn kernel_mult(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    block_m: usize,
    block_k: usize,
    block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    unsafe {
        if SIMD_WIDTH == block_n && SIMD_WIDTH == block_k {
            return kernel_mult_simd(x, y, t, block_m, s_x, s_y);
        }
    }

    kernel_mult_scalar(x, y, t, block_m, block_k, block_n, s_x, s_y);
}
