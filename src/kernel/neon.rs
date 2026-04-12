#[cfg(all(feature = "neon", target_arch = "aarch64"))]
use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32};

pub fn kernel_mult_simd(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_m: usize,
    _block_k: usize,
    _block_n: usize,
    s_x: usize,
    s_y: usize,
) {
    unsafe {
        let aptr = a.as_ptr();
        let bptr = b.as_ptr();
        let cptr = c.as_mut_ptr();
        let i_row = vld1q_f32(bptr);
        let ii_row = vld1q_f32(bptr.add(s_y));
        let iii_row = vld1q_f32(bptr.add(s_y * 2));
        let iv_row = vld1q_f32(bptr.add(s_y * 3));

        let mut aoffset = 0;
        let mut coffset = 0;
        for _ in 0..block_m {
            let arow = aptr.add(aoffset);
            let c_row = cptr.add(coffset);
            let mut acc = vld1q_f32(c_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*arow.add(0)), i_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*arow.add(1)), ii_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*arow.add(2)), iii_row);
            acc = vfmaq_f32(acc, vdupq_n_f32(*arow.add(3)), iv_row);
            vst1q_f32(c_row, acc1);
            aoffset += s_x;
            coffset += s_y;
        }
    }
}
