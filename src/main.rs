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

use stellar::algebra::bmethods::{diff_min, pack};
use stellar::arch::SIMD_WIDTH;
use stellar::kernel::matkerns::kernel_ut_mult;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;
// const MC: usize = 16;
// const PC: usize = 16;
// const NC: usize = 16;

thread_local! {
    static PACK: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((vec![0f32; MC * PC], vec![0f32; PC * NC], vec![0f32; MC * NC]));
}
pub fn tensor_ut_block(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    // suffix c: chunk, suffix a: actual
    let d_sub = if m > p { m - p } else { 0 };
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .enumerate()
        .for_each(|(lc_idx, (t, x))| {
            PACK.with(|workspace_cell| {
                let lc = lc_idx * MC;
                let d_add = lc; //
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let dy = PC * s_y;
                let (xend, mut yend, tend);
                let rows = x.len() / s_x;
                let ma = rows;
                (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_ut_contraction(
                            &x_pack,
                            &y_pack,
                            t_accum,
                            d_add,
                            d_sub + pc,
                            ma,
                            pa,
                            na,
                            PC,
                            NC,
                            NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_ut_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    mut d_add: usize,
    mut d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            // println!("xoffset {xoffset:?}");
            // println!("toffset {toffset:?}");
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..n).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
                // if d_sub + j + p >= d_add {
                if d_sub + p > d_add {
                    kernel_ut_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        // d_sub,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                }
            }
            toffset += dt;
            xoffset += dx;
            d_add += SIMD_WIDTH;
        }
    }
}
// #[cfg(test)]
// #[cfg(feature = "avx2")]
// mod test_upper_triangular_dispatch {
//     use super::*;
// {
    use stellar::algebra::ndmethods::basic_mult;
    use stellar::equality::approximate::approx_vector_eq;
    use stellar::random::generation::generate_random_matrix;
    use stellar::structure::ndarray::NdArray;
    // #[test]
    fn test_gemm_equivalence() {
        let ikj = [
            // (128, 128, 128)
            // (2, 12, 2),
            // (16, 16, 16),
            // (9, 16, 8),
            // (9, 16, 9),
            // (32, 32, 32),
            // (9, 8, 8),
            // (8, 1, 1),
            // (8, 8, 8),
            // (6, 4, 8),
            // (2, 2, 1),
            // (2, 9, 1),
            // (1, 1, 1),
            // (8, 9, 8),
            // (3, 9, 1),
            // (8, 8, 9),
            // (2, 9, 1),
            // (2, 10, 1),
            // (1, 9, 1),
            // (4, 8, 1),
            // (1, 2, 1),
            // (1, 1, 1),
            // (1, 8, 1),
            // (1, 1, 8),
            // (6, 4, 8),
            // (6, 8, 4),
            // (8, 4, 6),
            // (4, 8, 6),
            // (4, 6, 8),
            // (8, 6, 4),
            // (8, 8, 8),

            // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            // (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            // (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            // (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            // (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
            // (MC + 1, PC, NC + 1),
            // (MC + 1, PC, NC - 1),
            // (MC + 1, PC, NC),
            // (MC - 1, PC, NC),
            // (MC, PC + 1, NC),
            // (MC, PC - 1, NC),
            // (MC, PC, NC),
            // (256, 256, 256),
            // (256, 1024, 512),
            (512, 512, 512),
            // (1024, 64, 1024),
        ];
        for (i, k, j) in ikj {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            upper_equivalence_mkn(i, k, j);
        }
    }
    /// Case 1 / Case 2
    /// -------+---------
    /// * * *  / * * * *
    /// * * *  / 0 * * *
    /// 0 * *  /
    /// 0 0 *  /
    fn filter_upper_triangle(a: &mut NdArray) {
        // i - (m - n);
        let (m, n) = (a.dims[0], a.dims[1]);
        let data = &mut a.data;
        let mut d_sub = if m > n { m - n } else { 0 }; 
        let mut d_plus = 0;
        for i in 0..m {
            for j in 0..n {
                if j + d_sub >= d_plus { break; }
                data[i * n + j] = 0f32;
            }
            d_plus += 1;
        }
    }
    fn upper_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut x_base = x.clone();
        filter_upper_triangle(&mut x_base);
        // println!("x_base {x_base:?}");
        // println!("y_base {y:?}");
        let expected = basic_mult(&x_base, &y);
        let mut result = vec![0f32; m * n];
        tensor_ut_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        let inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}\nresult {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
// }

fn main() {
    test_gemm_equivalence();
    println!("success");
}
