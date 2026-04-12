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
// 4. AVX-512 blocksizes  ← 2hrs, great benchmark result
// 5. Trait refactor      ← important but least urgent
// use stellar::algebra::mmethods::tensor_kernel;
// use stellar::algebra::ndmethods::basic_mult;
// use stellar::arch::SIMD_WIDTH;
// use stellar::equality::approximate::approx_vector_eq;
// use stellar::random::generation::generate_random_matrix;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32};

// #[target_feature(enable = "avx,fma")]
#[cfg(feature = "aarch64")]
#[cfg(target_arch = "aarch64")]
pub fn kernel_mult_avx2(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    block_v: usize,
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
        for _ in 0..block_v {
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

fn main() {}
