use stellar::algebra::ndmethods::basic_mult;
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::decomposition::lower_upper::LuPivotDecompose;
use stellar::decomposition::schur::real_schur;
use stellar::equality::approximate::approx_vector_tol_eq;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;
const TOLERANCE: f32 = 1e-2;

fn determinant(x: NdArray, workspace: &mut [f32]) -> f32 {
    let (rows, cols) = (x.dims[0], x.dims[1]);
    let card = rows.min(cols);

    let lu = LuPivotDecompose::new(x, workspace);
    let mut det = 1f32;
    let lu_d = &lu.matrix.data;
    for k in 0..card {
        det *= lu_d[k * cols + k];
    }
    det.abs()
}
fn test_reconstruct() {
    // for n in 1..12 {
    //     check_reconstruct(n);
    // }
    // check_reconstruct(4);
    check_reconstruct(3);
}
fn check_reconstruct(n: usize) {
    let mut workspace = vec![0f32; n];
    let x = generate_random_matrix(n, n);
    let x = basic_mult(&x, &x.transpose());
    let det = determinant(x.clone(), &mut workspace);
    if det.abs() < TOLERANCE {
        println!("determinant to low\ndet{det:?}");
        return;
    }
    let kernel = x.clone();
    let nkernel = create_identity_matrix(n);
    let schur = real_schur(kernel, nkernel, &mut workspace);
    let q = schur.rotation;
    let q_star = q.transpose();
    println!("q {q:?}");
    let expect = basic_mult(&q, &x);
    let expect = basic_mult(&expect, &q_star);
    let result = schur.kernel;
    println!("expect {expect:?}");
    println!("result {result:?}");
    println!("determinant {det:?}");
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
    let mut workspace = vec![0f32; n];
    let schur = real_schur(kernel, nkernel, &mut workspace);
    let q = schur.rotation;
    let q_star = q.transpose();
    println!("q {q:?}");
    let expect = create_identity_matrix(n);
    let result = basic_mult(&q, &q_star);
    // println!("expect {expect:?}");
    // println!("result {result:?}");
    assert!(approx_vector_tol_eq(&result.data, &expect.data, TOLERANCE));
}
fn main() {
    test_reconstruct();
    // test_orthogonal();
}
