use stellar::decomposition::svd::bidiagonalization::{lbidiagonal, ubidiagonal};
use stellar::algebra::ndmethods::matrix_mult;
use stellar::decomposition::svd::bulge_chasing::{decomp_lgivens, decomp_ugivens};
use stellar::algebra::ndmethods::create_identity_vector;
use stellar::decomposition::svd::bidiagonalization::full_ubidiagonal;
use stellar::decomposition::svd::bulge_chasing::full_decomp_ugivens;
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

// A 4x4 identity matrix flattened into a single Vec<f32> (row-major order)
fn test() {
    let rows: usize = 3;
    let cols: usize = 3;
    let card: usize = rows.min(cols);
    // let rows: usize = 6;
    // let cols: usize = 4;

    let stride: usize = cols;
    let maximum = rows.max(cols);
    let mut w = vec![0f32; maximum];
    let mut p = vec![0f32; maximum];

    let mut b: Vec<f32> = generate_random_vector(rows * cols);
    // for i in 0..rows {
    //     for j in 0..=i {
    //     // for j in i+1..cols {
    //         b[i * cols + j] = 0f32;
    //     }
    // }
    let input = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    println!("before matrix {input:?}");

    ubidiagonal(&mut b, &mut p, &mut w, rows, cols, card, stride);
    // ubidiagonal(&mut b, &mut p, &mut w, rows, cols, stride);
    let bidiag = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    println!("after bidiag {bidiag:?}");
    decomp_ugivens(&mut b, card, stride, 40, 1e-10, 1e-8);

    let output = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("after rotations {output:?}");
}

fn main() {
    // let rows: usize = 3;
    // let cols: usize = 3;
    let rows: usize = 6;
    let cols: usize = 3;
    let card: usize = rows.min(cols);
    let mut u = create_identity_vector(rows, rows);
    let mut v = create_identity_vector(cols, cols);

    let stride: usize = cols;
    let maximum = rows.max(cols);
    let mut w = vec![0f32; maximum];
    let mut p = vec![0f32; maximum];

    let mut b: Vec<f32> = generate_random_vector(rows * cols);
    let input = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    println!("u {u:?}");
    println!("before matrix {input:?}");

    full_ubidiagonal(
        &mut b, &mut u, &mut v, &mut p, &mut w, rows, cols, card, stride,
    );
    let bidiag = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    let u = NdArray {
        dims: vec![rows, rows],
        data: u.clone(),
    };
    let v = NdArray {
        dims: vec![cols, cols],
        data: v.clone(),
    };

    println!("v {v:?}");

    println!("after bidiag {bidiag:?}");
    let check = matrix_mult(&u, &bidiag);
    let u_ortho = matrix_mult(&u, &u.transpose());
    let v_ortho = matrix_mult(&v, &v.transpose());
    
    println!("checking u_ortho {u_ortho:?}");
    println!("checking v_ortho {v_ortho:?}");
    
    // println!("after bidiag {bidiag:?}");
    let reconstruct = bidiag;

    let reconstruct = matrix_mult(&u, &reconstruct);
    let reconstruct = matrix_mult(&reconstruct, &v.transpose());
    println!("checking reconstruct {reconstruct:?}");
    // println!("checking u_ortho {u_ortho:?}");
    // full_decomp_ugivens(&mut b, &mut u, &mut v, rows, cols, card, stride, 40, 1e-10, 1e-8);

    // let output = NdArray {
    //     dims: vec![rows, cols],
    //     data: b.clone(),
    // };
    // println!("after rotations {output:?}");
}
