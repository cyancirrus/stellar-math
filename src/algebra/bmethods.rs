#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::kernel_mult;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// NOTE: could set these as cache sizes so threads reflect the amount of work
const LC: usize = 64;
const MC: usize = 64;
const PC: usize = 256;
const NC: usize = 128;

///  tensor_kernel
///  - accumulates the multiplication into the target matrix
///  - t += x * y
#[inline(always)]
pub fn tensor_kernel_new(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_contraction(&x.data, &y.data, target, m, p, n, p, n, n);
    } else {
        tensor_blockkern(&x.data, &y.data, target, m, p, n);
    }
}

///  tensor_kernel_into
///  # not accumulated
///   - returns t = x * y
#[inline(always)]
pub fn tensor_kernel_into(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    target.fill(0f32);
    tensor_kernel_new(x, y, target);
}

thread_local! {
    static PACK: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = RefCell::new((vec![0f32; MC * PC], vec![0f32; PC * NC], vec![0f32; MC * NC]));
}
pub fn tensor_blockkern(x_d: &[f32], y_d: &[f32], t_d: &mut [f32], m: usize, p: usize, n: usize) {
    // suffix c: chunk, suffix a: actual
    t_d.par_chunks_mut(LC * n)
        .zip(x_d.par_chunks(LC * p))
        .for_each(|(t, x)| {
            PACK.with(|workspace_cell| {
                let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
                let (mut xoffset, mut yoffset, mut toffset) = (0, 0, 0);
                let rows = x.len() / p;
                for mc in (0..rows).step_by(MC) {
                    let ma = (rows - mc).min(MC);
                    for nc in (0..n).step_by(NC) {
                        let na = (n - nc).min(NC);
                        yoffset = 0;
                        t_accum.fill(0f32);
                        for pc in (0..p).step_by(PC) {
                            let pa = (p - pc).min(PC);
                            pack(&y_d[yoffset + nc..yoffset + pa * n], y_pack, pa, na, NC, n);
                            pack(&x[xoffset + pc..xoffset + ma * p], x_pack, ma, pa, PC, p);
                            tensor_contraction(&x_pack, &y_pack, t_accum, ma, pa, na, PC, NC, NC);
                            yoffset += PC * n;
                        }
                        pack(
                            &t_accum,
                            &mut t[toffset + nc..toffset + ma * n],
                            ma,
                            na,
                            n,
                            NC,
                        );
                    }
                    xoffset += MC * p;
                    toffset += MC * n;
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
pub fn tensor_contraction(
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
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..n).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
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
                );
            }
            toffset += SIMD_WIDTH * s_t;
            xoffset += SIMD_WIDTH * s_x;
        }
    }
}
#[cfg(test)]
mod test_kernel_block {
    use crate::algebra::bmethods::{tensor_blockkern, tensor_contraction};
    use crate::algebra::ndmethods::basic_mult;
    use crate::arch::SIMD_WIDTH;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;

    #[test]
    fn test_outkern_equivalence() {
        let ikj = [
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (4, 6, 8),
            (4, 8, 6),
            (8, 4, 6),
            (8, 6, 4),
            (8, 8, 8),
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        ];
        for (i, k, j) in ikj {
            test_outkern_equivalence_mkn(i, k, j);
        }
    }
    fn test_outkern_equivalence_mkn(m: usize, p: usize, n: usize) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        tensor_contraction(&x.data, &y.data, &mut result, m, p, n, p, n, n);
        // let inspect = NdArray {
        //     dims: vec![m, n],
        //     data: result.clone(),
        // };
        // println!("expected {expected:?}");
        // println!("actual {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    #[test]
    fn test_gemm_equivalence() {
        let ikj = [
            (256, 256, 256),
            (1, 1, 1),
            (8, 1, 1),
            (1, 8, 1),
            (1, 1, 8),
            (6, 4, 8),
            (6, 8, 4),
            (4, 6, 8),
            (4, 8, 6),
            (8, 4, 6),
            (8, 6, 4),
            (8, 8, 8),
            (16, 16, 16),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
            (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
            (256, 1024, 512),
            (512, 512, 512),
            (1024, 64, 1024),
        ];
        for (i, k, j) in ikj {
            println!("(i: {i:?}, k: {k:?}, j: {j:})");
            test_blockkern_equivalence_mkn(i, k, j);
        }
    }
    fn test_blockkern_equivalence_mkn(m: usize, k: usize, n: usize) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        tensor_blockkern(&x.data, &y.data, &mut result, m, k, n);
        let inspect = NdArray {
            dims: vec![m, n],
            data: result.clone(),
        };
        // println!("expected {expected:?}");
        // println!("actual {inspect:?}");
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
}
