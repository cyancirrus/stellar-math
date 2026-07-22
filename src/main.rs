#![allow(unused_imports, dead_code, unused_variables, unused)]
use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::matrix_mult;
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use stellar::equality::approximate::approx_vector_tol_eq;
use stellar::random::generation::{
    generate_identity_vector, generate_random_matrix, generate_random_vector,
};
use stellar::structure::ndarray::NdArray;
use stellar::decomposition::francis:: primitives:: {
    full_hessenberg,
};

use stellar::decomposition::francis::symmetric:: {
decomp_sym,
    francis_iteration_sym,
};
use stellar::decomposition::francis::complex:: {
    decomp_cpx,
    francis_iteration_cpx,
    francis_iteration_cpx_2x2,
};
fn generate_symmetric_vector(n: usize) -> Vec<f32> {
    let a = generate_random_matrix(n, n);
    matrix_mult(&a, &a.transpose()).data
}
fn check_hessen_sym() {
    let (rows, cols) = (5, 5);
    let stride = 5;
    let mut h = generate_symmetric_vector(rows);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("after {kernel:?}");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    println!("rotation {rotation:?}");
    let ortho = matrix_mult(&rotation, &rotation.transpose());
    println!("ortho rr' {ortho:?}");
    let ortho = matrix_mult(&rotation.transpose(), &rotation);
    println!("ortho r'r {ortho:?}");
    let reconstruct = matrix_mult(&kernel, &rotation);
    let reconstruct = matrix_mult(&rotation.transpose(), &reconstruct);
    println!("reconstruct {reconstruct:?}");
}
fn check_hessen() {
    let (rows, cols) = (5, 5);
    let stride = 5;
    let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("after {kernel:?}");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    println!("rotation {rotation:?}");
    let ortho = matrix_mult(&rotation, &rotation.transpose());
    println!("ortho rr' {ortho:?}");
    let ortho = matrix_mult(&rotation.transpose(), &rotation);
    println!("ortho r'r {ortho:?}");
    let reconstruct = matrix_mult(&kernel, &rotation);
    println!("reconstruct {reconstruct:?}");
}
fn check_iteration_sym() -> NdArray {
    let c = 6;
    let (rows, cols) = (c, c);
    let stride = c;
    let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let s = c * c;
    let mut bl = s.saturating_sub(2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("hessenberg {kernel:?}");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    francis_iteration_sym(&mut h, rows, rows, stride, tl, bl);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_decomp_sym() -> NdArray {
    let c = 4;
    let (rows, cols) = (c, c);
    let stride = c;
    let mut h = generate_symmetric_vector(rows);
    // let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("hessenberg {kernel:?}");
    decomp_sym(&mut h, c, c, c);
    // francis_iteration(&mut h, rows, stride);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_iteration_cpx() -> NdArray {
    let c = 4;
    let (rows, cols) = (c, c);
    let stride = c;
    let s: usize = c * c;
    let mut bl = s.saturating_sub(2);
    let mut tl = s.saturating_sub(stride + 2);
    let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("hessenberg {kernel:?}");
    println!("-------------------------------");
    println!("-------------------------------");
    let rotation = NdArray {
        dims: vec![rows, cols],
        data: r.clone(),
    };
    let mut p = [0f32; 3];
    w.fill(0f32);
    francis_iteration_cpx(&mut h, &mut p, &mut w, rows, rows, stride, tl, bl);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    println!("final {output:?}");
    output
}
fn check_decomp_cpx() -> NdArray {
    let c = 6;
    let (rows, cols) = (c, c);
    let stride = c;
    let mut h = generate_random_vector(rows * cols);
    let mut r = generate_identity_vector(rows, cols);
    let mut p = vec![0f32; cols];
    let mut w = vec![0f32; rows];
    let input = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    // println!("before {input:?}");
    full_hessenberg(&mut h, &mut r, &mut p, &mut w, rows, cols, stride);
    let kernel = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    // println!("hessenberg {kernel:?}");
    w.fill(0f32);
    decomp_cpx(&mut h, &mut w, c, c, c);
    let output = NdArray {
        dims: vec![rows, cols],
        data: h.clone(),
    };
    // println!("final {output:?}");
    output
}

fn main() {
    check_decomp_sym();
    for i in 0..1000 {
        check_decomp_cpx();
        // println!("-----------------");
    }
    // check_iteration_cpx();
    // check_hessen_sym();
    // check_iteration_sym();
    // check_hessen();
    // test_orthogonal();
    // TODO
    // if range > 1 {

    //     let output = NdArray {
    //         dims: vec![size, size],
    //         data: h.to_vec(),
    //     };
    //     println!("-------------- error output inspect----------");
    //     println!("{output:?}");
    //     let (m00, m01) = (h[tl], h[tl + 1]);
    //     let (m10, m11) = (h[bl], h[bl + 1]);
    //     let d = (m00 - m11) / 2.0;
    //     let disc = d * d + m01 * m10;
    //     println!(
    //         "max_iter reached (r:{range}, e1: {}, e2: {}, disc: {disc:?})",
    //         h[e1], h[e2]
    //     );
    // }
}
