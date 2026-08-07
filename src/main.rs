use stellar::decomposition::svd::bidiagonalization::{decomp_ugivens, lbidiagonal, ubidiagonal};
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

// A 4x4 identity matrix flattened into a single Vec<f32> (row-major order)

fn main() {
    let rows: usize = 6;
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
    decomp_ugivens(&mut b, card, stride, 20, 1e-10, 1e-8);

    let output = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("after rotations {output:?}");
}
