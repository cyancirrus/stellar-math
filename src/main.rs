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

use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
use stellar::algebra::ndmethods::basic_mult;
use stellar::arch::SIMD_WIDTH;
use stellar::equality::approximate::approx_vector_eq;
use stellar::kernel::matkerns::{kernel_lt_mult, kernel_mult};
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// NOTE: could set these as cache sizes so threads reflect the amount of work
const LC: usize = 64;
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;

#[inline(always)]
fn diff_min(x: usize, b: usize, t: usize) -> usize {
    if x - b < t { x - b } else { t }
}
thread_local! {
    static PACK: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((vec![0f32; MC * PC], vec![0f32; PC * NC], vec![0f32; MC * NC]));
}
pub fn tensor_lt_block(
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
    t_d.par_chunks_mut(LC * n)
        .zip(x_d.par_chunks(LC * p))
        .enumerate()
        .for_each(|(lc_idx, (t, x))| {
            PACK.with(|workspace_cell| {
                let lc = lc_idx * LC;
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let (dx, dt, dy) = (MC * s_x, MC * s_t, PC * s_y);
                let (mut xend, mut yend, mut tend);
                let (mut xoffset, mut yoffset, mut toffset) = (0, 0, 0);
                let rows = x.len() / s_x;
                for mc in (0..rows).step_by(MC) {
                    let ma = diff_min(rows, mc, MC);
                    let t_bound = lc + mc + ma;
                    (xend, tend) = (ma * s_x, ma * s_t);
                    for nc in (0..n).step_by(NC) {
                        let na = diff_min(n, nc, NC);
                        t_accum.fill(0f32);
                        yoffset = 0;
                        for pc in (0..t_bound).step_by(PC) {
                            // for pc in (0..p).step_by(PC) {
                            let pa = diff_min(p, pc, PC);
                            yend = pa * s_y;
                            pack(&x[xoffset + pc..xoffset + xend], x_pack, ma, pa, PC, s_x);
                            pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                            tensor_lt_contraction(
                                &x_pack, &y_pack, t_accum, lc, pc, ma, pa, na, PC, NC, NC,
                            );
                            yoffset += dy;
                        }
                        // unpack
                        pack(
                            &t_accum,
                            &mut t[toffset + nc..toffset + tend],
                            ma,
                            na,
                            s_y,
                            NC,
                        );
                    }
                    xoffset += dx;
                    toffset += dt;
                }
            })
        });
}
/// # pack transfers a copy of data from d to pack
/// * to inverse simply exchange d and b
/// - d ~ M(r, s)
///
/// * d: contains the source data of x sliced to begin at mc
/// * b: contains the target pack for the outer iteration loop
/// * re: size of the r-block
/// * se: size of the s-block
/// * s_b: stride of block
/// * s_d: stride of the matrix d
#[inline(always)]
fn pack(d: &[f32], b: &mut [f32], re: usize, se: usize, s_b: usize, s_d: usize) {
    unsafe {
        let mut doffset = 0;
        let mut boffset = 0;
        for _ in 0..re {
            b.get_unchecked_mut(boffset..boffset + se)
                .copy_from_slice(&d.get_unchecked(doffset..doffset + se));
            boffset += s_b;
            doffset += s_d;
        }
    }
}
pub fn tensor_lt_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    g_i: usize,
    g_k: usize,
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
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..=i).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
                if g_i + i == g_k {
                    kernel_lt_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                // } else {
                } else if g_i + i >= g_k {
                    kernel_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
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
        }
    }
}
fn test_gemm_equivalence() {
    let ikj = [
        // (256, 256, 256),
        // (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
        // (1, 1, 1),
        // (8, 1, 1),
        // (1, 8, 1),
        // (1, 1, 8),
        // (6, 4, 8),
        // (6, 8, 4),
        // (4, 6, 8),
        // (4, 8, 6),
        // (8, 4, 6),
        // (8, 6, 4),
        (8, 8, 8),
        (16, 16, 16),
        // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        // (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
        // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
        // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        // (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
        // (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
        // (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        // (256, 1024, 512),
        // (512, 512, 512),
        // (1024, 64, 1024),
    ];
    for (i, k, j) in ikj {
        println!("(i: {i:?}, k: {k:?}, j: {j:})");
        test_lower_equivalence_mkn(i, k, j);
    }
}
fn filter_lower_triangle(a: &mut NdArray) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    for i in 0..rows {
        for j in i + 1..cols {
            d[i * cols + j] = 0f32;
        }
    }
}
fn test_lower_equivalence_mkn(m: usize, p: usize, n: usize) {
    let x = generate_random_matrix(m, p);
    let y = generate_random_matrix(p, n);
    let mut x_base = x.clone();
    filter_lower_triangle(&mut x_base);
    let expected = basic_mult(&x_base, &y);
    let mut result = vec![0f32; m * n];
    tensor_lt_block(&x.data, &y.data, &mut result, m, p, n, m, p, n);
    let inspect = NdArray {
        dims: vec![m, n],
        data: result.clone(),
    };
    println!("expected {expected:?}");
    println!("actual {inspect:?}");
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}

fn main() {
    test_gemm_equivalence();
}
