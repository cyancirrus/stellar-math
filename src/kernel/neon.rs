#[cfg(all(feature = "neon", target_arch = "aarch64"))]
use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32};

pub fn kernel_mult_simd(
    mut xptr: *const f32,
    mut yptr: *const f32,
    mut tptr: *mut f32,
    block_m: usize,
    s_x: usize,
    s_y: usize,
) {
    unsafe {
        let i_row = vld1q_f32(bptr);
        let ii_row = vld1q_f32(bptr.add(s_y));
        let iii_row = vld1q_f32(bptr.add(s_y * 2));
        let iv_row = vld1q_f32(bptr.add(s_y * 3));

        for _ in 0..block_m {
            let mut acc = vld1q_f32(c_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*xptr.add(0)), i_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*xptr.add(1)), ii_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*xptr.add(2)), iii_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*xptr.add(3)), iv_row);
            vst1q_f32(tptr, acc1);
            xptr = xptr.add(s_x);
            tptr = tptr.add(s_y);
        }
    }
}
