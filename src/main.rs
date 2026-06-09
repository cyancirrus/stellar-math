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
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
use stellar::algebra::bmethods::{diff_min, pack};
use stellar::arch::SIMD_WIDTH;
use stellar::kernel::matkerns::{kernel_rlt_mult, kernel_ut_mult};
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;

thread_local! {
    static PACK: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((vec![0f32; MC * PC], vec![0f32; PC * NC], vec![0f32; MC * NC]));
}
pub fn tensor_rlt_block(
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
    let d_add = n - n.min(p) + 1;
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .enumerate()
        .for_each(|(mc_idx, (t, x))| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                // let d_add = d_add + mc_idx * MC;
                let dy = PC * s_y;
                let ma = x.len() / s_x;
                let (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    let mut yoffset = 0;
                    let d_add = d_add + nc * PC;
                    for pc in (0..p).step_by(PC) {
                        let pa = diff_min(p, pc, PC);
                        let yend = pa * s_y;
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        tensor_rlt_contraction(
                            &x_pack, &y_pack, t_accum, d_add, pc, ma, pa, na, PC, NC, NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
                }
            })
        });
}
pub fn tensor_rlt_contraction(
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
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        // if d_add + ii_end > d_sub {
        for j in (0..n).step_by(SIMD_WIDTH) {
            println!("y {:?}", &y_d[j..j + 10]);
            let mut xoffset = 0;
            let mut toffset = 0;
            let jj_end = SIMD_WIDTH.min(n - j);
            // println!("hello");
            // println!("j {j:?}");
            // if d_add + jj_end > d_sub {
            if d_add + jj_end + m > d_sub {
                for i in (0..m).step_by(SIMD_WIDTH) {
                    let ii_end = SIMD_WIDTH.min(m - i);
                    kernel_rlt_mult(
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
                    );
                    toffset += dt;
                    xoffset += dx;
                }
            } else {
                println!("d_add {d_add:?}, jj_end {jj_end:?}, d_sub {d_sub:?}");
            }
            d_sub += SIMD_WIDTH;
        }
    }
}
use stellar::algebra::ndmethods::basic_mult;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
fn test_gemm_equivalence() {
    let ikj = [
        (8, 8, 9),
        
        (1, 1, 8),
        (1, 8, 1),
        (6, 4, 8),
        (8, 8, 8),
        (2, 2, 1),
        (1, 1, 1),
        (8, 9, 8),
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

        (9, 16, 8),
        (9, 16, 9),
        (32, 32, 32),
        (16, 16, 16),
        (1, 9, 1),
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
        // (512, 512, 512),
        // (1024, 64, 1024),
    ];
    for (i, k, j) in ikj {
        println!("(i: {i:?}, k: {k:?}, j: {j:})");
        rlower_equivalence_mkn(i, k, j);
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
fn rlower_equivalence_mkn(m: usize, p: usize, n: usize) {
    let x = generate_random_matrix(m, p);
    let y = generate_random_matrix(p, n);
    let mut y_base = y.clone();
    println!("y_base {y_base:?}");
    filter_lower_triangle(&mut y_base);
    println!("x_base {x:?}");
    let expected = basic_mult(&x, &y_base);
    let mut result = vec![0f32; m * n];
    tensor_rlt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
    let _inspect = NdArray {
        dims: vec![m, n],
        data: result.clone(),
    };
    println!("expected {expected:?}");
    println!("actual {_inspect:?}");
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}
fn main() {
    test_gemm_equivalence();
    println!("success");
}
