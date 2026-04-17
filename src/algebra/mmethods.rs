#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::kernel_mult;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
const MC: usize = 32; // l2 cachesize
const PC: usize = 64; // l1 cachesize
const NC: usize = 256; // to be tuned
// const MC: usize = 64; // l2 cachesize
// const PC: usize = 256; // l1 cachesize
// const NC: usize = 1024; // to be tuned

thread_local! {
    static PROC_WORKSPACEX: RefCell<Vec<f32>> = RefCell::new(vec![0.0f32; SIMD_WIDTH * SIMD_WIDTH]);
    static PROC_WORKSPACEY: RefCell<Vec<f32>> = RefCell::new(vec![0.0f32; SIMD_WIDTH * SIMD_WIDTH]);
}

///  tensor_kernel
///  - accumulates the multiplication into the target matrix
///  - t += x * y
#[inline(always)]
pub fn tensor_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    if m <= MINIKERN_GATE && p <= MINIKERN_GATE << 1 && n <= MINIKERN_GATE {
        tensor_minikern(&x.data, &y.data, target, m, p, n)
    } else {
        tensor_parkern(&x.data, &y.data, target, m, p, n);
    }
}

///  tensor_kernel into
///   - returns x * y
#[inline(always)]
pub fn tensor_kernel_into(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    target.fill(0f32);
    tensor_kernel(x, y, target);
}

pub fn tensor_blockkern(x_d: &[f32], y_d: &[f32], t_d: &mut [f32], m: usize, p: usize, n: usize) {
    // NOTE: par at this level so that each core gets its own data
    // n p m is recommended
    for nc in (0..n).step_by(NC) {
        for pc in (0..p).step_by(PC) {
            // pack_x  ~ MC x PC; // to fit in l2
            for mc in (0..m).step_by(MC) {
                // pack_y ~ PC X NC; // to fit in l1
                // tensor_parkern(x_pack, y_pack, t_slice);
            }
        }
    }
}

/// # pack_x returns a panel of the original matrix x
/// - d ~ M(r, s)
///
/// * d: contains the source data of x sliced to begin at mc
/// * pack: contains the reused pack for the outer iteration loop
/// * rl: r-index in the outer loop for the begin
/// * re: size of the r-block
/// * sl: used for when there exists a partial block
/// * se: size of the s-block
/// * sd: stride of the matrix d
fn pack(
    d: &[f32],
    pack: &mut [f32],
    rl: usize,
    re: usize,
    sl: usize,
    se: usize,
    block: usize,
    stride: usize,
) {
    unsafe {
        // window offset
        let mut woffset = 0;
        // data offset
        let mut doffset = rl;
        for _ in 0..re {
            pack.get_unchecked_mut(woffset..woffset + se)
                .copy_from_slice(&d.get_unchecked(doffset..doffset + se));
            woffset += block;
            doffset += stride;
        }
    }
}

// /// # pack_x returns a panel of the original matrix x
// /// - x ~ M(m, p)
// ///
// /// * x_d: contains the source data of x sliced to begin at mc
// /// * x_pack: contains the reused pack for the outer iteration loop
// /// * mc: i-index in the outer loop for the begin
// /// * pc: k-index in the outer loop for the begin
// /// * end: used for when there exists a partial block
// /// * s_x: the iteration for the unsliced source matrix x
// fn pack_x(x_d: &[f32], x_pack:&mut [f32], mc:usize, pc:usize, end:usize, sx: usize) {
//     unsafe {
//         let mut xoffset = 0;
//         let mut woffset = pc;
//         for _ in 0..MC {
//             x_pack
//                 .get_unchecked_mut(woffset..woffset + end)
//                 .copy_from_slice(
//                     &x_d.get_unchecked(xoffset..xoffset + end),
//                 );
//             woffset += MC;
//             xoffset += sx;
//         }
//     }
// }

// /// # pack_y returns a panel of the original matrix y
// /// - y ~ M(p, n)
// ///
// /// * x_d: contains the source data of x sliced to begin at mc
// /// * x_pack: contains the reused pack for the outer iteration loop
// /// * mc: i-index in the outer loop for the begin
// /// * pc: k-index in the outer loop for the begin
// /// * end: used for when there exists a partial block
// /// * s_x: the iteration for the unsliced source matrix x
// fn pack_y(y_d: &[f32], y_pack:&mut [f32], pc:usize, nc:usize, end:usize, sy: usize) {
//     unsafe {
//         let mut yoffset = 0;
//         let mut woffset = pc;
//         for _ in 0..PC {
//             y_pack
//                 .get_unchecked_mut(woffset..woffset + end)
//                 .copy_from_slice(
//                     &y_d.get_unchecked(yoffset..yoffset + end),
//                 );
//             woffset += MC;
//             yoffset += sy;
//         }
//     }

// }

pub fn tensor_parkern(x_d: &[f32], y_d: &[f32], t_d: &mut [f32], _: usize, p: usize, n: usize) {
    unsafe {
        // will reuse allocation if available
        t_d.par_chunks_mut(SIMD_WIDTH * n)
            .zip(x_d.par_chunks(SIMD_WIDTH * p))
            .for_each(|(t_block_row, x_block_row)| {
                PROC_WORKSPACEX.with(|workspace_cell| {
                    let ii_end = x_block_row.len() / p;
                    let mut work_x = workspace_cell.borrow_mut();
                    let mut yoffset = 0;
                    for k in (0..p).step_by(SIMD_WIDTH) {
                        let kk_end = SIMD_WIDTH.min(p - k);
                        let mut xoffset = k;
                        let mut woffset = 0;
                        for _ in 0..ii_end {
                            work_x
                                .get_unchecked_mut(woffset..woffset + kk_end)
                                .copy_from_slice(
                                    &x_block_row.get_unchecked(xoffset..xoffset + kk_end),
                                );
                            woffset += SIMD_WIDTH;
                            xoffset += p;
                        }
                        for j in (0..n).step_by(SIMD_WIDTH) {
                            let jj_end = SIMD_WIDTH.min(n - j);
                            kernel_mult(
                                &work_x,
                                y_d.get_unchecked(yoffset + j..),
                                t_block_row.get_unchecked_mut(j..),
                                ii_end,
                                kk_end,
                                jj_end,
                                SIMD_WIDTH,
                                n,
                            );
                        }
                        yoffset += SIMD_WIDTH * n;
                    }
                })
            });
    }
}
pub fn tensor_minikern(x_d: &[f32], y_d: &[f32], t_d: &mut [f32], m: usize, p: usize, n: usize) {
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
                        p,
                        n,
                    );
                }
                yoffset += SIMD_WIDTH * n;
            }
            toffset += SIMD_WIDTH * n;
            xoffset += SIMD_WIDTH * p;
        }
    }
}

#[cfg(test)]
mod test_cached_matrix_methods {
    use super::*;
    use crate::algebra::ndmethods::basic_mult;
    use crate::equality::approximate::approx_vector_eq;
    use crate::random::generation::generate_random_matrix;

    #[test]
    // #[cfg(feature = "avx2")]
    fn test_par_kernel_equivalence() {
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
            (16, 8, 16),
        ];
        let mut result = vec![0f32; 16 * 16];
        for (i, k, j) in ikj {
            result.fill(0f32);
            test_par_kernel_equivalence_mpn(i, k, j, &mut result);
        }
    }
    fn test_par_kernel_equivalence_mpn(m: usize, p: usize, n: usize, result: &mut [f32]) {
        let x = generate_random_matrix(m, p);
        let y = generate_random_matrix(p, n);
        let expected = basic_mult(&x, &y);
        tensor_parkern(&x.data, &y.data, &mut result[..m * n], m, p, n);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
    #[test]
    fn test_minikern_equivalence() {
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
            test_minikern_equivalence_mkn(i, k, j);
        }
    }
    fn test_minikern_equivalence_mkn(m: usize, k: usize, n: usize) {
        let x = generate_random_matrix(m, k);
        let y = generate_random_matrix(k, n);
        let mut result = vec![0f32; m * n];
        let expected = basic_mult(&x, &y);
        tensor_minikern(&x.data, &y.data, &mut result, m, k, n);
        assert!(approx_vector_eq(&expected.data, &result[..m * n]));
    }
}
