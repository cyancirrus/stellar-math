// use stellar::algebra::ndmethods::basic_mult;
// use stellar::algebra::ndmethods::create_identity_matrix;
// use stellar::decomposition::lq::AutumnDecomp;
// use stellar::decomposition::schur::real_schur;
// use stellar::equality::approximate::{approx_scalar_eq, approx_vector_eq, approx_vector_tol_eq};
// use stellar::random::generation::generate_random_matrix;
// use stellar::structure::ndarray::NdArray;

// // LQ -> QL

// fn check_ql(n: usize) {
//     let x = generate_random_matrix(n, n);
//     let mut workspace = vec![f32::NAN; n];
//     let mut explicit = create_identity_matrix(n);
//     // let mut implicit = vec![f32::NAN; n * n];
//     let mut implicit = generate_random_matrix(n, n );
//     let lq = AutumnDecomp::new(x);
//     lq.left_apply_l(&mut explicit, &mut workspace);
//     lq.left_apply_q(&mut explicit, &mut workspace);
//     lq.ql_apply(&mut implicit, &mut workspace);
//     let expect = explicit;
//     let result = implicit.clone();
//     assert!(approx_vector_eq(&result.data, &expect.data,), "result {result:?}\nexpect {expect:?}");
// }
// fn test_n_autumn_reconstruct(n: usize) {
//     let a = generate_random_matrix(n, n);
//     let expected = a.clone();
//     let mut workspace = vec![f32::NAN; n];
//     let autumn = AutumnDecomp::new(a.clone());
//     println!("tau {:?}\nlq {:?}", autumn.t, autumn.h);

//     // let mut i = create_identity_matrix(n);
//     // autumn.right_apply_l(&mut i);
//     // autumn.right_apply_q(&mut i, &mut workspace);
//     // assert!(approx_vector_eq(&i.data, &expected.data));
//     let mut i = create_identity_matrix(n);
//     autumn.left_apply_q(&mut i, &mut workspace);
//     autumn.left_apply_l(&mut i, &mut workspace);
//     assert!(approx_vector_eq(&i.data, &expected.data));
//     println!("i {i:?}\nexpected {expected:?}");
//     // let mut i = create_identity_matrix(n);
//     // autumn.left_apply_l(&mut i, &mut workspace);
//     // autumn.right_apply_q(&mut i, &mut workspace);
//     // assert!(approx_vector_eq(&i.data, &expected.data));
//     // let mut i = create_identity_matrix(n);
//     // autumn.left_apply_lt(&mut i, &mut workspace);
//     // autumn.left_apply_qt(&mut i, &mut workspace);
//     // i.transpose_square();
//     // assert!(approx_vector_eq(&i.data, &expected.data));
//     // let mut i = create_identity_matrix(n);
//     // autumn.right_apply_qt(&mut i);
//     // autumn.right_apply_lt(&mut i);
//     // i.transpose_square();
//     // assert!(approx_vector_eq(&i.data, &expected.data));
// }

fn main() {
    // test_n_autumn_reconstruct(2);
    // check_ql(6);
    // test_orthogonal();
    // test_reconstruction_3();
}
