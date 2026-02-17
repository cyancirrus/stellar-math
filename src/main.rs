#![allow(dead_code, unused_imports)]

use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::lower_upper::LuPivotDecomp;
use stellar::equality::approximate::approx_vector_eq;
use stellar::random::generation::{generate_random_matrix, generate_random_vector};
use stellar::structure::ndarray::NdArray;

// fn test_reconstruct() {
//     let n = 4;
//     let x = generate_random_matrix(n, n);
//     let lu = LuPivotDecomp::new(x.clone());
//     // let lu = LuPivotDecomp::new_dl(x.clone());
//     // let lu = LuDecomposition::new(x.clone());
//     let out = lu.reconstruct();
//     assert_eq!(x, out);
// }

// PA ~ LU
//
// LUx = PLUx

// f32 epsilon
const EPSILON: f32 = 1e-6;
pub fn approx_condition_vector_eq(a: &[f32], b: &[f32], k:&f32) -> bool {
    let mut diff_norm = 0_f32;
    let mut target_norm = 0_f32;
    let n = a.len();
    for i in 0..n {
        diff_norm += (a[i] - b[i]).abs();
        target_norm += b[i].abs();
    }
    (diff_norm / target_norm.max(f32::MIN_POSITIVE)) <= EPSILON * k
}

fn solve_inplace_vec_ax_y(a:NdArray, y:&mut [f32]) -> bool {
    // LU x = y;
    // find x then ensure equals y
    let lu = LuPivotDecomp::new(a);
    let expected = y.to_vec();
    // lu.pivot_inplace_vec(&mut expected);
    let result = y;
    let condition = lu.condition();
    lu.solve_inplace_vec(result);
    lu.left_apply_u_vec( result);
    lu.left_apply_l_vec( result);
    lu.unpivot_inplace_vec(result);
    // println!("expected {expected:?}, result {result:?}");
    approx_condition_vector_eq(&expected, &result, &condition)
}
fn test_random_solve_inplace_vec_ax_y() -> (usize, usize) {
    let dimensions = vec![ 2, 3, 4, 7, 23];
    let mut success = 0;
    for n in dimensions {
        let x = generate_random_matrix(n, n);
        let mut y = generate_random_vector(n);
        if solve_inplace_vec_ax_y(x.clone(), &mut y) {
            success += 1;
        }
    }
    (success, 5)


}
fn main() {
    // test_reconstruct();
    let mut success = 0;
    let mut n = 0;
    for _ in 0..10_000 {
        let (s, k) = test_random_solve_inplace_vec_ax_y();
        success += s;
        n += k;
    }
    println!("Final success metric {:?}", (success as f32) / (n as f32));
}

// fn solve_inplace_vec_ax_y(a:NdArray, y:&mut [f32]) {
//     // LU x = y;
//     // find x then ensure equals y
//     let lu = LuPivotDecomp::new_test(a);
//     let expected = y.to_vec();
//     // lu.pivot_inplace_vec(&mut expected);
//     let result = y;
//     let condition = lu.condition();
//     lu.solve_inplace_vec(result);
//     lu.left_apply_u_vec( result);
//     lu.left_apply_l_vec( result);
//     lu.unpivot_inplace_vec(result);
//     // println!("expected {expected:?}, result {result:?}");
//     assert!(approx_condition_vector_eq(&expected, &result, &condition));
// }
// fn test_random_solve_inplace_vec_ax_y() {
//     let dimensions = vec![ 2, 3, 4, 7, 23];
//     for n in dimensions {
//         let x = generate_random_matrix(n, n);
//         let mut y = generate_random_vector(n);
//         solve_inplace_vec_ax_y(x.clone(), &mut y);
//     }
// }
