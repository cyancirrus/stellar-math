use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::schur::real_schur;
use stellar::equality::approximate::{approx_vector_tol_eq};
use stellar::random::generation::generate_random_matrix;
const TOLERANCE:f32 = 1e-3;
fn test_reconstruct() {
    for n in 1..12 {
        check_reconstruct(n);
    }
}
fn check_reconstruct(n: usize) {
    let x = generate_random_matrix(n, n);
    let kernel = x.clone();
    let nkernel = create_identity_matrix(n);
    let mut workspace = vec![0f32;n];
    let schur = real_schur(kernel, nkernel, &mut workspace);
    let q = schur.rotation;
    let q_star = q.transpose();
    println!("q {q:?}");
    let expect = basic_mult(&q, &x);
    let expect = basic_mult(&expect, &q_star);
    let result = schur.kernel;
    println!("expect {expect:?}");
    println!("result {result:?}");
    assert!(approx_vector_tol_eq(&result.data, &expect.data, TOLERANCE));
}
fn test_orthogonal() {
    for n in 1..12 {
        check_orthogonal(n);
    }
}
fn check_orthogonal(n: usize) {
    let x = generate_random_matrix(n, n);
    let kernel = x.clone();
    let nkernel = create_identity_matrix(n);
    let mut workspace = vec![0f32;n];
    let schur = real_schur(kernel, nkernel, &mut workspace);
    let q = schur.rotation;
    let q_star = q.transpose();
    println!("q {q:?}");
    let expect = create_identity_matrix(n);
    let result = basic_mult(&q, &q_star);
    println!("expect {expect:?}");
    println!("result {result:?}");
    assert!(approx_vector_tol_eq(&result.data, &expect.data,TOLERANCE));
}
fn main() {
    test_reconstruct();
    test_orthogonal();
}
