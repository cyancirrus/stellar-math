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

// 0. solve packing
// 1. Animate demo        ← most legible to employers
// 2. Blog redesign       ← makes everything else findable
// 3. Triangle kernel     ← 2hrs, unblocks LQ block
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
use stellar::algebra::bmethods::{diff_min, pack};
use stellar::arch::SIMD_WIDTH;
use stellar::kernel::matkerns::{kernel_rut_mult, kernel_tut_mult, kernel_ut_mult};
// // DEBUG PARAMS
// const MC: usize = 8;
// const PC: usize = 16;
// const NC: usize = 8;

// const MC: usize = 8;
// const PC: usize = 8;
// const NC: usize = 8;
//
// const MC: usize = 32;
// const PC: usize = 24;
// const NC: usize = 16;
// // PROD PARAMS
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;

thread_local! {
    static PACK: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((vec![0f32; MC * PC], vec![0f32; PC * NC], vec![0f32; MC * NC]));
}
pub fn tensor_tut_block(
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
    // diagonal
    // suffix c: chunk, suffix a: actual
    // let d_add = p - p.min(m) + 1;
    let d_sub = m.saturating_sub(p);
    t_d.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(mc_idx, t)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let d_add = mc_idx * MC;
                let dy = PC * s_y;
                let d_xt = PC * s_x;
                let ma = diff_min(m, mc_idx * MC, MC);
                // let (xend, tend) = (ma * s_x, ma * s_t);
                let tend = ma * s_t;
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    // base column offset
                    let mut xoffset = mc_idx * MC;
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        pack(&x_d[xoffset..], x_pack, pa, ma, MC, s_x);
                        pack(&y_d[yoffset + nc..], y_pack, pa, na, NC, s_y);
                        tensor_tut_contraction(
                            &x_pack,
                            &y_pack,
                            t_accum,
                            d_add,
                            d_sub + pc,
                            ma,
                            pa,
                            na,
                            MC,
                            NC,
                            NC,
                        );
                        yoffset += dy;
                        xoffset += d_xt;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_tut_contraction(
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
        let dx = SIMD_WIDTH;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            // if d_add + ii_end > d_sub {
            if d_sub + p > d_add {
                for j in (0..n).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(n - j);
                    kernel_tut_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
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
            // toffset += dt;
            // xoffset += dx;
            // d_sub += SIMD_WIDTH;
        }
    }
}
use stellar::algebra::ndmethods::basic_mult;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
fn test_gemm_equivalence() {
    let ikj = [
        (9, 2, 2),
        (9, 8, 8),
        (9, 16, 9),
        (9, 16, 8),
        (9, 16, 9),
        (1, 1, 1),
        (8, 8, 8),
        (1, 2, 1),
        (1, 1, 8),
        (4, 4, 4),
        (6, 4, 6),
        (2, 2, 1),
        (8, 8, 8),
        (8, 18, 16),
        (8, 33, 16),
        (8, 10, 8),
        (8, 8, 10),
        (8, 16, 8),
        (1, 8, 1),
        (4, 20, 2),
        (4, 20, 2),
        (4, 24, 4),
        (8, 9, 8),
        (8, 9, 8),
        (6, 4, 8),
        (2, 2, 1),
        (3, 9, 1),
        (4, 8, 1),
        (1, 2, 1),
        (8, 1, 1),
        (6, 8, 4),
        (8, 4, 6),
        (4, 8, 6),
        (4, 6, 8),
        (8, 6, 4),
        (2, 9, 1),
        (2, 10, 1),
        (1, 9, 1),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        (1, 25, 9),
        (32, 64, 32),
        (32, 32, 32),
        (1, 26, 10),
        (9, 16, 9),
        (16, 16, 16),
        (16, 32, 16),
        (MC + 1, PC, NC + 1),
        (MC + 1, PC, NC - 1),
        (MC + 1, PC, NC),
        (MC - 1, PC, NC),
        (MC, PC + 1, NC),
        (MC, PC - 1, NC),
        (MC, PC, NC),
        (128, 512, 32),
        (32, 512, 32),
        (256, 1024, 512),
        (256, 256, 256),
        (256, 1024, 512),
        (512, 512, 512),
        (1024, 64, 1024),
        (16, 32, 16),
    ];
    for (i, k, j) in ikj {
        println!("(i: {i:?}, k: {k:?}, j: {j:})");
        ltu_equivalence_mkn(i, k, j);
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
            if j + d_sub >= d_plus {
                break;
            }
            data[i * n + j] = 0f32;
        }
        d_plus += 1;
    }
}
/// * * * * * * * 0 0 0
/// * * * * * * * * 0 0
/// * * * * * * * * * 0
/// * * * * * * * * * *
fn filter_lower_triangle(a: &mut NdArray) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let t = cols.min(rows);
    let s = rows.saturating_sub(cols);
    // don't remove from last row
    for i in 1..t {
        for j in 0..i {
            d[(rows - i - s) * cols - j - 1] = 0f32;
        }
    }
}
fn ltu_equivalence_mkn(m: usize, p: usize, n: usize) {
    let y = generate_random_matrix(p, n);
    let mut x = generate_random_matrix(m, p);
    let mut x_base = x.clone();
    // filter_lower_triangle(&mut x_base);
    filter_upper_triangle(&mut x_base);
    x.transpose_inplace(); // stored in transpose
    // println!("x_base {x_base:?}");
    // println!("y {y:?}");
    let expected = basic_mult(&x_base, &y);
    let mut result = vec![0f32; m * n];
    // m, n, n b/c X is s tored in it's transposed state
    tensor_tut_block(&x.data, &y.data, &mut result, m, p, n, m, n, n);
    let _inspect = NdArray {
        dims: vec![m, n],
        data: result.clone(),
    };
    // println!("expected {expected:?}");
    // println!("actual {_inspect:?}");
    assert!(
        approx_vector_eq(&expected.data, &result[..m * n]),
        "FAILURE WAS ({m:}, {p:}, {n:})"
    );
}
use rayon::ThreadPoolBuilder;
fn main() {
    // ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .expect("Failed to initialize global thread pool");
    test_gemm_equivalence();
    println!("success");
}
