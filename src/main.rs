#![allow(unused_imports, dead_code, unused_variables, unused)]
use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::matrix_mult;
use stellar::decomposition::francis::primitives::hessenberg;
use stellar::decomposition::francis::verify::{full_decomp_cpx, full_decomp_sym, full_hessenberg};
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::sgivens::{
    apply_g_left, apply_g_right, apply_gt_left, apply_gt_right, implicit_givens_rotation,
};
use stellar::equality::approximate::{approx_vector_eq, approx_vector_tol_eq};
use stellar::random::generation::{
    generate_approx_symmetric_vector, generate_identity_vector, generate_random_matrix,
    generate_random_vector,
};
use stellar::structure::ndarray::NdArray;

use stellar::decomposition::francis::complex::{
    decomp_cpx, francis_iteration_cpx, francis_iteration_cpx_2x2,
};
use stellar::decomposition::francis::symmetric::{decomp_sym, francis_iteration_sym};
fn main() {
}
