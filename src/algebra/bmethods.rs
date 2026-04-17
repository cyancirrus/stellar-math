#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::algebra::mmethods::{tensor_minikern, tensor_parkern};
use crate::kernel::matkerns::kernel_mult;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// const MC: usize = 32; // l2 cachesize
// const PC: usize = 64; // l1 cachesize
// const NC: usize = 256; // to be tuned
// const MC: usize = 64; // l2 cachesize
// const PC: usize = 256; // l1 cachesize
// const NC: usize = 1024; // to be tuned
const MC: usize = 128; // l2 cachesize
const PC: usize = 64; // l1 cachesize
const NC: usize = 1024; // to be tuned

thread_local! {
    static PACK_X: RefCell<Vec<f32>> = RefCell::new(vec![0f32; MC * PC]);
    static PACK_Y: RefCell<Vec<f32>> = RefCell::new(vec![0f32; PC * NC]);
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
    // TODO: remove this once threaded correctly
    let mut x_pack = vec![0f32; MC * PC];
    let mut y_pack = vec![0f32; PC * NC];
    // suffix c: chunk, suffix a: actual
    for nc in (0..n).step_by(NC) {
        let na = (n - nc).min(NC);
        for pc in (0..p).step_by(PC) {
            // shared dimension
            let pa = (p - pc).min(PC);
            pack(&y_d[nc * n..], &mut y_pack, pa, na, NC, n);
            // pack_x  ~ MC x PC; // to fit in l2
            for mc in (0..m).step_by(MC) {
                let ma = (m - mc).min(MC);
                pack(&x_d[mc * p..], &mut x_pack, ma, pa, PC, p);
                // should i pass in a stride?
                tensor_minikern(&x_pack, &y_pack, &mut t_d[mc * n + nc..], ma, pa, na);
            }
        }
    }
}

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
                    );
                }
                yoffset += SIMD_WIDTH * s_y;
            }
            toffset += SIMD_WIDTH * s_t;
            xoffset += SIMD_WIDTH * s_x;
        }
    }
}

