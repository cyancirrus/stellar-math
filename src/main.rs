use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::schur::real_schur;
use stellar::equality::approximate::{approx_scalar_eq, approx_vector_eq, approx_vector_tol_eq};
use stellar::decomposition::qr::QrDecomposition;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

const TOLERANCE: f32 = 1e-3;

fn test_reconstruction_3() {
    let dims = vec![3, 3];
    let data = vec![-1.0, 2.0, 3.0, 5.0, 2.0, 4.0, -3.0, 0.7, 1.2];
    let x = NdArray::new(dims.clone(), data.clone());
    let schur = real_schur(x.clone());
    // let q = &schur.rotation;
    // let q_star = &schur.rotation.transpose();
    let q = &schur.rotation;
    let q_star = &schur.rotation.transpose();

    // let expect = basic_mult(&x, q);
    // let expect = basic_mult(q_star, &expect);
    let expect = basic_mult(q_star, &x);
    let expect = basic_mult(&expect, q);
    let result = schur.kernel;
    println!("expect {expect:?}");
    println!("result {result:?}");
    assert!(approx_vector_tol_eq(&expect.data, &result.data, TOLERANCE));
}
fn test_orthogonal() {
    for n in 1..12 {
        check_orthogonal(n);
    }
}
fn check_orthogonal(n: usize) {
    let x = generate_random_matrix(n, n);
    let schur = real_schur(x);
    let q = schur.rotation;
    let q_star = q.transpose();
    // println!("q {q:?}");
    let expect = create_identity_matrix(n);
    let result = basic_mult(&q, &q_star);
    // println!("expect {expect:?}");
    // println!("result {result:?}");
    assert!(approx_vector_eq(&result.data, &expect.data,));
}
fn check_lq(n: usize) {
    let x = generate_random_matrix(n, n);
    let explicit = QrDecomposition::new(x.clone());
    let mut inline = QrDecomposition::new(x.clone());
    let rq_explicit = basic_mult(&explicit.triangle, &explicit.projection_matrix());
    inline.triangle_rotation();
    let expect = rq_explicit;
    let result = &inline.triangle;
    println!("expect {expect:?}");
    println!("result {result:?}");
    assert!(approx_vector_eq(&result.data, &expect.data,));
}

fn main() {
    check_lq(3);
    // test_orthogonal();
    // test_reconstruction_3();
}
