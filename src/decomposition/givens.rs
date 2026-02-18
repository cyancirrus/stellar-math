#![allow(warnings)]
use crate::algebra::ndmethods::{create_identity_matrix, matrix_mult, transpose};
use crate::decomposition::qr;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

const CONVERGENCE_CONDITION: f32 = 1e-6;

pub struct SingularValueDecomp {
    pub u: NdArray,
    pub s: NdArray,
    pub v: NdArray,
}

impl SingularValueDecomp {
    pub fn new(u: NdArray, s: NdArray, v: NdArray) -> Self {
        Self { s, u, v }
    }
}

// HACK: Fix as soon as possible there's a weird sign in my rotations
pub fn full_givens_iteration(mut u:NdArray, mut s: NdArray, mut v: NdArray) -> SingularValueDecomp {
    // takes in bidiagonal and returns full SVD
    let m = s.dims[0];
    let n = s.dims[1];
    let k = m.min(n);
    // row-space, column-space
    let mut max_iteration = 1 << 8;
    // left work
    while offdiag_norm(&s) > CONVERGENCE_CONDITION && max_iteration > 0 {
        for i in 0..k - 1 {
            // TODO: Optimize, there's a better way to do this it's only a trace over a bidiagonal
            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[(i + 1) * n + i]);
            // below diagonal element
            let g = embed_givens(m, i, i + 1, cosine, sine);
            let g_t = g.transpose();
            s = matrix_mult(&g, &s);
            u = matrix_mult(&u, &g_t);

            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[i * n + i + 1]);
            let g = embed_givens(n, i, i + 1, cosine, sine);
            let g_t = g.transpose();
            s = matrix_mult(&s, &g_t);
            v = matrix_mult(&v, &g_t);
        }
        max_iteration -= 1
    }
    SingularValueDecomp { u, s, v }
}


pub fn givens_iteration(mut s: NdArray) -> SingularValueDecomp {
    // takes in bidiagonal and returns full SVD
    let m = s.dims[0];
    let n = s.dims[1];
    let k = m.min(n);
    // row-space, column-space
    let mut u = create_identity_matrix(m);
    let mut v = create_identity_matrix(n);
    let mut max_iteration = 1 << 8;
    // left work
    while offdiag_norm(&s) > CONVERGENCE_CONDITION && max_iteration > 0 {
        for i in 0..k - 1 {
            // TODO: Optimize, there's a better way to do this it's only a trace over a bidiagonal
            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[(i + 1) * n + i]);
            // below diagonal element
            let g = embed_givens(m, i, i + 1, cosine, sine);
            let g_t = g.transpose();
            s = matrix_mult(&g, &s);
            u = matrix_mult(&u, &g_t);

            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[i * n + i + 1]);
            let g = embed_givens(n, i, i + 1, cosine, sine);
            let g_t = g.transpose();
            s = matrix_mult(&s, &g_t);
            v = matrix_mult(&v, &g_t);
        }
        max_iteration -= 1
    }
    SingularValueDecomp { u, s, v }
}

fn embed_givens(n: usize, i: usize, j: usize, c: f32, s: f32) -> NdArray {
    let mut array = create_identity_matrix(n);
    array.data[i * n + i] = c;
    array.data[i * n + j] = s;
    array.data[j * n + i] = -s;
    array.data[j * n + j] = c;
    array
}

// m x n, m x m x n

fn offdiag_norm(s: &NdArray) -> f32 {
    let m = s.dims[0];
    let n = s.dims[1];
    let mut norm = 0.0;
    for i in 0..m.min(n) - 1 {
        // upper diagonal
        // norm += s.data[i * n + i + 1].abs();
        norm += s.data[i * n + i + 1].abs() + s.data[(i + 1) * n + 1].abs();
    }
    norm
}

pub fn implicit_givens_rotation(a: f32, b: f32) -> (f32, f32, f32) {
    let t: f32;
    let tt: f32;
    let s: f32;
    let c: f32;
    let r: f32;

    if a == 0_f32 {
        c = 0_f32;
        s = 1_f32;
        r = b;
    } else if b.abs() > a.abs() {
        t = a / b;
        tt = (1_f32 + t.powi(2)).sqrt();
        s = 1_f32 / tt;
        c = s * t;
        r = b * tt;
    } else {
        t = b / a;
        tt = (1_f32 + t.powi(2)).sqrt();
        c = 1_f32 / tt;
        s = c * t;
        r = a * tt;
    }
    let r: f32 = (a.powi(2) + b.powi(2)).sqrt();
    (r, c, s)
}
