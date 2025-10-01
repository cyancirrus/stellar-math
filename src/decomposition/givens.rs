#![allow(warnings)]
use crate::algebra::ndmethods::{create_identity_matrix, tensor_mult, transpose};
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
            s = tensor_mult(2, &g, &s);
            u = tensor_mult(2, &u, &g);

            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[i * n + i + 1]);
            let g = embed_givens(n, i, i + 1, cosine, sine);
            let g_t = g.transpose();
            s = tensor_mult(2, &s, &g_t);
            v = tensor_mult(2, &v, &g);
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

// use stellar::decomposition::svd::golub_kahan;
// use stellar::decomposition::schur::real_schur;
// use stellar::decomposition::qr::qr_decompose;
// use stellar::decomposition::givens::givens_iteration;
// use stellar::structure::ndarray::NdArray;
//
// fn main() {
//     // {
//         // Eigen values 2, -1
//         let mut data = vec![0_f32; 4];
//         let dims = vec![2; 2];
//         data[0] = -1_f32;
//         data[1] = 0_f32;
//         data[2] = 5_f32;
//         data[3] = 2_f32;
//     // }
//     // {
//     //     data = vec![0_f32; 9];
//     //     dims = vec![3; 2];
//     //     data[0] = 1_f32;
//     //     data[1] = 2_f32;
//     //     data[2] = 3_f32;
//     //     data[3] = 3_f32;
//     //     data[4] = 4_f32;
//     //     data[5] = 5_f32;
//     //     data[6] = 6_f32;
//     //     data[7] = 7_f32;
//     //     data[8] = 8_f32;
//     // }
//     let x = NdArray::new(dims, data.clone());
//     println!("x: {:?}", x);
//     //
//     let reference = golub_kahan(x.clone());
//     println!("Reference {:?}", reference);

//     let y = qr_decompose(x.clone());
//     println!("triangle {:?}", y.triangle);

//     let real_schur = real_schur(x.clone());
//     println!("real schur kernel {:?}", real_schur.kernel);

//     let svd = givens_iteration(reference);
//     println!("svd u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd.u, svd.s, svd.v);

// }
