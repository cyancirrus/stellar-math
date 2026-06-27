use crate::arch::SIMD_WIDTH;
use crate::structure::ndarray::NdArray;
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
///  - zero out t if u don't wish for accumulation
///
///   stride is always how the data is stored not a matrix dimension
#[inline(always)]
pub fn tensor_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    stride_kernel(&x.data, &y.data, target, m, p, n, p, n, n);
}
#[inline(always)]
pub fn tensor_tkernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[0], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[1], y.dims[0], y.dims[1]);
    stride_tkernel(&x.data, &y.data, target, m, p, n, m, n, n);
}
#[inline(always)]
pub fn tensor_lt_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (p - p.min(m) + 1, 0);
    stride_lt_kernel(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
}
#[inline(always)]
pub fn tensor_ut_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (0, m.saturating_sub(p));
    stride_ut_kernel(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
}
#[inline(always)]
pub fn tensor_rlt_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (n - n.min(p), 0);
    stride_rlt_kernel(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
}
#[inline(always)]
pub fn tensor_rut_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[1], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[0], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (p - p.min(n) + 1, 0);
    stride_rut_kernel(&x.data, &y.data, target, d_add, d_sub, m, p, n, p, n, n);
}
#[inline(always)]
pub fn tensor_tlt_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[0], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[1], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (p - p.min(m) + 1, 0);
    stride_tlt_kernel(&x.data, &y.data, target, d_add, d_sub, m, p, n, m, n, n);
}
#[inline(always)]
pub fn tensor_tut_kernel(x: &NdArray, y: &NdArray, target: &mut [f32]) {
    debug_assert_eq!(x.dims[0], y.dims[0], "inner dimension mismatch");
    let (m, p, n) = (x.dims[1], y.dims[0], y.dims[1]);
    let (d_add, d_sub) = (0, m.saturating_sub(p));
    stride_tut_kernel(&x.data, &y.data, target, d_add, d_sub, m, p, n, m, n, n);
}
#[inline(always)]
fn assert_stride_bounds(
    b_x: usize,
    b_y: usize,
    b_t: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    debug_assert!( b_x <= s_x, "invaid x-stride length");
    debug_assert!( b_y <= s_y, "invaid y-stride length");
    debug_assert!( b_t <= s_t, "invaid t-stride length");
}
#[inline(always)]
fn assert_stride_capacity(
    x: &[f32],
    y: &[f32],
    t: &[f32],
    r_x: usize,
    r_y: usize,
    r_t: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    debug_assert!( r_x * s_x <= x.len(), "valid x-vector not large enough for dims");
    debug_assert!( r_y * s_y <= y.len(), "invalid y-vector not large enough for dims");
    debug_assert!( r_t * s_t <= t.len(), "invalid t-vector not large enough for dims");
}
#[rustfmt::skip]
#[inline(always)]
pub fn stride_kernel(x: &[f32], y: &[f32], t: &mut [f32], m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(p, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, m, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_contraction(x, y, t, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_block(x, y, t, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_tkernel(x: &[f32], y: &[f32], t: &mut [f32], m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(m, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, p, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_tcontraction(x, y, t, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_tblock(x, y, t, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_lt_kernel(x: &[f32], y: &[f32], t: &mut [f32], d_add:usize, d_sub:usize, m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(p, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, m, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_lt_contraction(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_lt_block(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_ut_kernel(x: &[f32], y: &[f32], t: &mut [f32], d_add:usize, d_sub:usize, m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(p, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, m, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_ut_contraction(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_ut_block(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_rlt_kernel(x: &[f32], y: &[f32], t: &mut [f32], d_add:usize, d_sub:usize, m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(p, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, m, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_rlt_contraction(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_rlt_block(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_rut_kernel(x: &[f32], y: &[f32], t: &mut [f32], d_add:usize, d_sub:usize, m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(p, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, m, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_rut_contraction(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_rut_block(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_tlt_kernel(x: &[f32], y: &[f32], t: &mut [f32], d_add:usize, d_sub:usize, m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(m, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, p, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_tlt_contraction(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_tlt_block(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    }
}
#[inline(always)]
pub fn stride_tut_kernel(x: &[f32], y: &[f32], t: &mut [f32], d_add:usize, d_sub:usize, m: usize, p: usize, n: usize, s_x: usize, s_y: usize, s_t: usize) {
    #[cfg(debug_assertions)]
    assert_stride_bounds(m, n, n, s_x, s_y, s_t);
    #[cfg(debug_assertions)]
    assert_stride_capacity(x, y, t, p, p, m, s_x, s_y, s_t);
    if m <= MINIKERN_GATE && n <= MINIKERN_GATE {
        tensor_tut_contraction(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    } else {
        tensor_tut_block(x, y, t, d_add, d_sub, m, p, n, s_x, s_y, s_t);
    }
}
