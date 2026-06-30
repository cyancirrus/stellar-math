use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::schur::real_schur;
use stellar::equality::approximate::{approx_scalar_eq, approx_vector_eq, approx_vector_tol_eq};
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

// LQ -> QL

fn check_ql(n: usize) {
    let x = generate_random_matrix(n, n);
    let mut workspace = vec![f32::NAN; n];
    let mut explicit = create_identity_matrix(n);
    let mut implicit = vec![f32::NAN; n * n];
    // let mut implicit = generate_random_matrix(n, n );
    let lq = AutumnDecomp::new(x);
    lq.mat_left_apply_l(&mut explicit, &mut workspace);
    lq.mat_left_apply_q(&mut explicit, &mut workspace);
    lq.ql_apply(&mut implicit, &mut workspace, n, n);
    let expect = explicit;
    let result = NdArray { dims: vec![n, n], data: implicit.clone() };
    assert!(approx_vector_eq(&result.data, &expect.data,), "result {result:?}\nexpect {expect:?}");
}
fn main() {
    check_ql(6);
    // test_orthogonal();
    // test_reconstruction_3();
}
