#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use crate::kernel::avx2::kernel_mult_simd;
#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
use crate::kernel::avx512::kernel_mult_simd;
#[cfg(not(any(feature = "avx2", feature = "avx512", feature = "neon")))]
use crate::kernel::default::kernel_mult_simd;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
use crate::kernel::neon::kernel_mult_simd;

#[inline(never)]
pub fn kernel_mult(
    x: &[f32],
    y: &[f32],
    t: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        kernel_mult_simd(
            x.as_ptr(),
            y.as_ptr(),
            t.as_mut_ptr(),
            m,
            p,
            n,
            s_x,
            s_y,
            s_t,
        );
    }
}
