#![allow(dead_code, unused_imports)]

use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::random::generation::{generate_random_matrix, generate_random_vector};
use stellar::structure::ndarray::NdArray;
use stellar::equality::approximate::{approx_vector_eq, approx_condition_eq};


fn solve_inplace_ax_y(a:NdArray, y:&mut NdArray) -> bool {
    // PA = LUx
    // Ax = y;
    // PAx = Py
    // LUx = Py;
    // => P'Py; when applying the compare
    let lu = LuPivotDecompose::new(a);
    let condition = lu.condition();
    let expected = y.data.clone();
    lu.solve_inplace(y);
    lu.left_apply_u( y);
    lu.left_apply_l( y);
    lu.unpivot_inplace(y);
    approx_condition_eq(&expected, &y.data, &condition)
}
fn random_solve_inplace_ax_y(dim:usize) -> bool {
    let x = generate_random_matrix(dim, dim);
    let mut y = generate_random_matrix(dim, dim);
    solve_inplace_ax_y(x, &mut y) 
}

fn test_matrix_solve_accuracy() {
    let dims= vec![ 2, 3, 4, 7, 23];
    let k = dims.len();
    let n = 200;
    let den= k * n;
    let mut num= 0;
    for d in dims {
        for _ in 0..n {
            num += random_solve_inplace_ax_y(d) as usize;
        }
    }
    assert!( num as f32 / den as f32 > 0.99);
}

fn main() {
    test_matrix_solve_accuracy();
}



// fn test_solve_inplace_ax_y(a:NdArray, y:&mut [f32]) -> bool {
//     // PA = LUx
//     // Ax = y;
//     // PAx = Py
//     // LUx = Py;
//     // => P'Py; when applying the compare
//     let lu = LuPivotDecompose::new(a);
//     let expected = y.to_vec();
//     let result = y;
//     let condition = lu.condition();
//     lu.solve_inplace_vec(result);
//     lu.left_apply_u_vec( result);
//     lu.left_apply_l_vec( result);
//     lu.unpivot_inplace_vec(result);
//     approx_condition_vector_eq(&expected, &result, &condition)
// }
// fn test_random_solve_inplace_ax_y(dim:usize) -> bool {
//     let x = generate_random_matrix(dim, dim);
//     let mut y = generate_random_vector(dim);
//     test_solve_inplace_ax_y(x.clone(), &mut y) 
// }

// fn test_vector_solve_accuracy() {
//     let dims= vec![ 2, 3, 4, 7, 23];
//     let k = dims.len();
//     let n = 200;
//     let den= k * n;
//     let mut num= 0;
//     for d in dims {
//         for _ in 0..n {
//             num += test_random_solve_inplace_ax_y(d) as usize;
//         }
//     }
//     assert!( num as f32 / den as f32 > 0.99);
// }
