use stellar::decomposition::svd::bidiagonalization::{bidiagonal, decomp_givens};
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

// A 4x4 identity matrix flattened into a single Vec<f32> (row-major order)

fn main() {
    let rows: usize = 4;
    let cols: usize = 4;
    let stride: usize = 4;
    let mut w = vec![0f32; 4];
    let mut p = vec![0f32; 4];

    let mut b: Vec<f32> = generate_random_vector(rows * cols);
    let input = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    println!("before matrix {input:?}");

    bidiagonal(&mut b, &mut p, &mut w, rows, cols, stride);
    let bidiag = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };

    println!("after bidiag {bidiag:?}");
    decomp_givens(&mut b, rows, cols, stride, 20, 1e-10, 1e-8);

    let output = NdArray {
        dims: vec![rows, cols],
        data: b.clone(),
    };
    println!("after rotations {output:?}");
}
