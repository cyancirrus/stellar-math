#![allow(warnings)]
use crate::algebra::ndmethods::{create_identity_matrix, tensor_mult, transpose};
use crate::decomposition::qr;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

struct GivensDecomposition {
    kernel: NdArray,
    rotation: NdArray,
}

impl GivensDecomposition {
    pub fn new(kernel: NdArray, rotation: NdArray) -> Self {
        Self { kernel, rotation }
    }
}

fn givens_rotation(a: f32, b: f32) -> NdArray {
    // TODO: Ensure that we actually provide r instead of explicit calculation
    let mut givens = vec![0_f32; 4];

    let t: f32;
    let s: f32;
    let c: f32;

    if b.abs() > a.abs() {
        // t = 2_f32 *a/b;
        t = 1_f32 * a / b;
        s = 1_f32 / (1_f32 + t.powi(2)).sqrt();
        c = s * t;
    } else {
        // t = 2_f32 * b/a;
        t = 1_f32 * b / a;
        c = 1_f32 / (1_f32 + t.powi(2)).sqrt();
        s = c * t;
    }
    givens[0] = c;
    givens[1] = s;
    givens[2] = -s;
    givens[3] = c;
    NdArray::new(vec![2; 2], givens)
}

fn givens_iteration(first: bool, mut givens: GivensDecomposition) -> GivensDecomposition {
    let rows = givens.kernel.dims[0];
    let cols = givens.kernel.dims[1];
    let rotation: NdArray;
    let left_rotation = givens_rotation(givens.kernel.data[0], givens.kernel.data[1]);
    // // if !first {
    givens.kernel = tensor_mult(2, &left_rotation, &givens.kernel);
    //     println!("upper cancel {:?}", givens.kernel);
    // // }
    let right_rotation = transpose(left_rotation.clone());
    // let right_rotation  = givens_rotation(givens.kernel.data[1], -givens.kernel.data[3]);
    givens.kernel = tensor_mult(2, &givens.kernel, &right_rotation);
    println!("re-pivot {:?}", givens.kernel);
    let ortho_check = tensor_mult(2, &left_rotation, &right_rotation);
    println!("Ortho check in givens {:?}", ortho_check);
    givens
}

fn givens_decomp(mut kernel: NdArray) -> GivensDecomposition {
    println!("Kernel {:?}", kernel);
    let mut upper = true;
    let rotation = create_identity_matrix(kernel.dims[0]);
    let mut givens = GivensDecomposition { rotation, kernel };
    let mut iteration = 8;
    let mut first = true;
    while iteration > 0 {
        iteration -= 1;
        upper = !upper;
        givens = givens_iteration(first, givens);
        println!("Givens iteration {:?}", givens.kernel);
        first = false;
    }
    givens
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
    (r, s, c)
}
