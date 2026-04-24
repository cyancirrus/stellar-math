#![allow(unused)]
use crate::algebra::mmethods::{tensor_minikern, tensor_parkern};
use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::kernel_mult;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// const LC: usize = 32; // l2 cachesize
// const MC: usize = 16; // l2 cachesize
// const PC: usize = 8; // l1 cachesize
// const NC: usize = 512; // to be tuned

// const LC: usize = 16; // l2 cachesize
// const MC: usize = 16; // l2 cachesize
// const PC: usize = 1024; // l1 cachesize
// const NC: usize = 128; // to be tuned

// const LC: usize = 16; // l2 cachesize
// const MC: usize = 16; // l2 cachesize
// const PC: usize = 512; // l1 cachesize
// const NC: usize = 128; // to be tuned

// THIS IS FOR THE TARGET CACHE
// const LC: usize = 64; // l2 cachesize
// const MC: usize = 64; // l2 cachesize
// const PC: usize = 128; // l1 cachesize
// const NC: usize = 128; // to be tuned

const LC: usize = 64; // l2 cachesize
const MC: usize = 64; // l2 cachesize
const PC: usize = 16; // l1 cachesize
const NC: usize = 128; // to be tuned

///  tensor_kernel
///  - accumulates the multiplication into the target matrix
///  - t += x * y
#[inline(always)]
pub fn tensor_kernel_new(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    if m <= MINIKERN_GATE && p <= MINIKERN_GATE << 1 && n <= MINIKERN_GATE {
        tensor_minikern(&x.data, &y.data, target, m, p, n)
    } else {
        tensor_blockkern(&x.data, &y.data, target, m, p, n);
    }
}

///  tensor_kernel into
///   - returns x * y
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
                let rows = x.len() / p;
                for mc in (0..rows).step_by(MC) {
                    let ma = (rows - mc).min(MC);
                    for nc in (0..n).step_by(NC) {
                        let na = (n - nc).min(NC);
                        t_accum.fill(0f32);
                        for pc in (0..p).step_by(PC) {
                            let pa = (p - pc).min(PC);
                            pack(&y_d[pc * n + nc..], y_pack, pa, na, NC, n);
                            pack(&x[mc * p + pc..], x_pack, ma, pa, PC, p);
                            tensor_newkern(&x_pack, &y_pack, t_accum, ma, pa, na, PC, NC, NC);
                        }
                        for k in 0..ma {
                            let trow = &t_accum[k * NC..k * NC + na];
                            let tout = &mut t[mc * n + k * n + nc..mc * n + k * n + nc + na];
                            tout.copy_from_slice(trow);
                        }
                    }
                }
            })
        });
}

// TODO: do the t pack
// pub fn tensor_blockkern(x_d: &[f32], y_d: &[f32], t_d: &mut [f32], m: usize, p: usize, n: usize) {
//     // suffix c: chunk, suffix a: actual
//     t_d.par_chunks_mut(LC * n)
//         .zip(x_d.par_chunks(LC * p))
//         .for_each(|(t, x)| {
//             PACK.with(|workspace_cell| {
//                 let (x_pack, y_pack, t_accum) = &mut *workspace_cell.borrow_mut();
//                 let rows = x.len() / p;
//                 for nc in (0..n).step_by(NC) {
//                     let na = (n - nc).min(NC);
//                     for pc in (0..p).step_by(PC) {
//                         let pa = (p - pc).min(PC);
//                         pack(&y_d[pc * n + nc..], y_pack, pa, na, NC, n);
//                         for mc in (0..rows).step_by(MC) {
//                             let ma = (rows - mc).min(MC);
//                             pack(&x[mc * p + pc..], x_pack, ma, pa, PC, p);
//                             tensor_newkern(
//                                 &x_pack,
//                                 &y_pack,
//                                 &mut t[mc * n + nc..],
//                                 ma,
//                                 pa,
//                                 na,
//                                 PC,
//                                 NC,
//                                 n,
//                             );
//                         }
//                     }
//                 }
//             })
//         });
// }

/// # pack_x returns a panel of the original matrix x
/// - d ~ M(r, s)
///
/// * d: contains the source data of x sliced to begin at mc
/// * pack: contains the reused pack for the outer iteration loop
/// * re: size of the r-block
/// * se: size of the s-block
/// * sd: stride of the matrix d
fn pack(d: &[f32], pack: &mut [f32], re: usize, se: usize, block: usize, stride: usize) {
    unsafe {
        let mut woffset = 0;
        let mut doffset = 0;
        for _ in 0..re {
            pack.get_unchecked_mut(woffset..woffset + se)
                .copy_from_slice(&d.get_unchecked(doffset..doffset + se));
            woffset += block;
            doffset += stride;
        }
    }
}

pub fn tensor_newkern(
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
        let mut yoffset;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            yoffset = 0;
            for k in (0..p).step_by(SIMD_WIDTH) {
                let kk_end = SIMD_WIDTH.min(p - k);
                for j in (0..n).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(n - j);
                    kernel_mult(
                        x_d.get_unchecked(xoffset + k..),
                        y_d.get_unchecked(yoffset + j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        ii_end,
                        kk_end,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    );
                }
                yoffset += SIMD_WIDTH * s_y;
            }
            toffset += SIMD_WIDTH * s_t;
            xoffset += SIMD_WIDTH * s_x;
        }
    }
}

#[cfg(test)]
mod test_kernel_block {
    use crate::algebra::bmethods::tensor_blockkern;
    use crate::algebra::ndmethods::basic_mult;
    use crate::arch::SIMD_WIDTH;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;
    use crate::structure::ndarray::NdArray;

    #[test]
    fn test_blockkern_equivalence() {
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
            // (512, 512, 512),
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
            // println!("(i: {i:?}, k: {k:?}, {j:})");
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
