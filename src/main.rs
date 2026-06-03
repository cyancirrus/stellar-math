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
use std::arch::x86_64::{
    _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_broadcast_ss, _mm256_castpd_ps,
    _mm256_castps_pd, _mm256_fmadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_mask_load_ps,
    _mm256_permute2f128_ps, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    _mm256_unpackhi_pd, _mm256_unpackhi_ps, _mm256_unpacklo_pd, _mm256_unpacklo_ps,
};
use std::cell::RefCell;
use stellar::algebra::ndmethods::basic_mult;
use stellar::arch::SIMD_WIDTH;
use stellar::equality::approximate::approx_vector_eq;
#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use stellar::kernel::avx2::constants::MASK;
use stellar::kernel::matkerns::{kernel_lt_mult, kernel_mult};
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// NOTE: could set these as cache sizes so threads reflect the amount of work
// const LC: usize = 64;
// const MC: usize = 64;
// const PC: usize = 256;
// const NC: usize = 128;
const LC: usize = 8;
const MC: usize = 8;
const PC: usize = 8;
const NC: usize = 8;

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
    // //println!("s_x {s_x:}, s_y: {s_y:}, s_t: {s_t:}");
    // suffix c: chunk, suffix a: actual
    let d_0 = (p - (p.min(m) - 1)) as isize;
    // //println!("d_0 : {d_0:}");
    t_d.par_chunks_mut(MC * n)
        .zip(x_d.par_chunks(MC * p))
        .enumerate()
        .for_each(|(lc_idx, (t, x))| {
            PACK.with(|workspace_cell| {
                let lc = lc_idx * LC;
                let d = d_0 + lc as isize;
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let dy = PC * s_y;
                let (xend, mut yend, tend);
                let rows = x.len() / s_x;
                let ma = rows;
                (xend, tend) = (ma * s_x, ma * s_t);
                for nc in (0..n).step_by(NC) {
                    let na = diff_min(n, nc, NC);
                    t_accum.fill(0f32);
                    // //println!("t_accum {t_accum:?}");
                    let mut yoffset = 0;
                    for pc in (0..p).step_by(PC) {
                        // //println!("stepping!");
                        let pa = diff_min(p, pc, PC);
                        yend = pa * s_y;
                        // //println!("x_pack before {x_pack:?}");
                        pack(&x[pc..xend], x_pack, ma, pa, PC, s_x);
                        pack(&y_d[yoffset + nc..yoffset + yend], y_pack, pa, na, NC, s_y);
                        // //println!("x_pack after {x_pack:?}");
                        tensor_lt_contraction(
                            &x_pack, &y_pack, t_accum, lc, pc, d, ma, pa, na, PC, NC, NC,
                        );
                        yoffset += dy;
                    }
                    // unpack
                    pack(&t_accum, &mut t[nc..tend], ma, na, s_t, NC);
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
    mut d: isize,
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
        // //println!("---------------------------");
        // //println!("p {p:}, d {d:}");
        // //println!("---------------------------");
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..n).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
                //println!("i {i:}, ii {ii_end:}, g_k {g_k:}, p: {p:}, j: {j:}, jj: {jj_end:}");
                if d + (ii_end as isize) > g_k as isize + 1 {
                    kernel_lt_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d as isize - g_k as isize,
                        ii_end,
                        // p.min((d as usize).wrapping_sub(g_k) ),
                        p.min(d as usize + ii_end),
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                } else {
                    // //println!("early exit");
                }
            }
            toffset += dt;
            xoffset += dx;
            d += SIMD_WIDTH as isize;
        }
    }
}
fn test_gemm_equivalence() {
    let ikj = [
        (32, 32, 32),
        // (16, 16, 16),
        // (8, 8, 9),
        // (3, 9, 1),
        // (2, 9, 1),
        // (2, 2, 1),
        // (2, 9, 1),
        // (2, 10, 1),
        // (1, 9, 1),
        // (4, 8, 1),
        // (1, 2, 1),
        // (1, 1, 1),
        // (8, 1, 1),
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
        // (256, 256, 256),
        // (256, 1024, 512),
        // (512, 512, 512),
        // (1024, 64, 1024),
    ];
    for (i, k, j) in ikj {
        println!("(i: {i:?}, k: {k:?}, j: {j:})");
        test_lower_equivalence_mkn(i, k, j);
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
fn test_lower_equivalence_mkn(m: usize, p: usize, n: usize) {
    let x = generate_random_matrix(m, p);
    let y = generate_random_matrix(p, n);
    let mut x_base = x.clone();
    filter_lower_triangle(&mut x_base);
    // //println!("x_base {x_base:?}");
    let expected = basic_mult(&x_base, &y);
    let mut result = vec![0f32; m * n];
    tensor_lt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
    // tensor_lt_block(&x.data, &y.data, &mut result, m, p, n, p, n, n);
    let inspect = NdArray {
        dims: vec![m, n],
        data: result.clone(),
    };
    //println!("y {y:?}");
    //println!("expected {expected:?}");
    //println!("actual {inspect:?}");
    assert!(approx_vector_eq(&expected.data, &result[..m * n]));
}

// macro for simd pack unrolling/pack_simd
fn main() {
    // let mut d_mat = generate_random_matrix(8, 64);
    // let mut d= d_mat.data.as_mut_ptr();
    // let mut b = vec![0f32; MC * PC].as_mut_ptr();
    // assert!(PC % SIMD_WIDTH == 0);
    // unsafe {
    //     pack_simd!(MC, PC, d, b, PC, 64);
    // }

    test_gemm_equivalence();
    println!("success");
}
