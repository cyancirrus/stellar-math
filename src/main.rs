#![allow(dead_code, unused_imports)]

use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::lower_upper::LuPivotDecomp;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

fn test_reconstruct() {
    let n = 4;
    let x = generate_random_matrix(n, n);
    let lu = LuPivotDecomp::new(x.clone());
    // let lu = LuPivotDecomp::new_dl(x.clone());
    // let lu = LuDecomposition::new(x.clone());
    let out = lu.reconstruct();
    assert_eq!(x, out);
}

fn main() {
    test_reconstruct();
}
