#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::kernel::matkerns::{kernel_mult, kernel_tmult};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
#[rustfmt::skip]
use crate::algebra::bmethods::contractions::{
    tensor_contraction,
    tensor_tcontraction,
    tensor_lt_contraction,
    tensor_ut_contraction,
    tensor_rlt_contraction,
    tensor_rut_contraction,
    tensor_tlt_contraction,
    tensor_tut_contraction,
};
#[rustfmt::skip]
use crate::algebra::bmethods::blocks:: {
    tensor_block,
    tensor_tblock,
    tensor_lt_block,
    tensor_ut_block,
    tensor_rlt_block,
    tensor_rut_block,
    tensor_tlt_block,
    tensor_tut_block,
};

const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;

///  tensor_kernel
///  - accumulates the multiplication into the target matrix
///  - t += x * y
///  tensor_kernel_into
///  # not accumulated
///   - returns t = x * y

#[inline(always)]
pub fn tensor_kernel_into(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    target.fill(0f32);
    tensor_kernel(x, y, target);
}

#[inline(always)]
pub fn tensor_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_contraction(&x.data, &y.data, target, m, p, n, p, n, n);
    } else {
        tensor_block(&x.data, &y.data, target, m, p, n, p, n, n);
    }
}
