#![allow(warnings)]
use StellarMath::algebra::{math, simd};
use StellarMath::decomposition::{qr, schur, householder};
use StellarMath::decomposition::householder::{householder_params, HouseholderReflection};
use StellarMath::decomposition::svd::golub_kahan_lanczos;
use StellarMath::decomposition::bidiagonal::{bidiagonal_qr, fast_bidiagonal_qr};
use StellarMath::structure::ndarray::NdArray;
use StellarMath::algebra::ndmethods::{
    transpose,
    tensor_mult,
    create_identity_matrix,
};
use rand::Rng;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn explicit_golub_kahan(mut a:NdArray) -> NdArray{
    let rows = a.dims[0];
    let cols = a.dims[1];
    let mut householder: HouseholderReflection = HouseholderReflection::new(0_f32, vec![0_f32;0]);

    let mut new:NdArray = create_identity_matrix(rows);
    // println!("Should be identity {:?}", new);

    for o in 0..cols.min(rows) - 1 {
        new = create_identity_matrix(rows);
        let mut column_vector = (o..rows).into_par_iter().map(|r| a.data[r*cols + o]).collect::<Vec<f32>>();
        householder = householder_params(&column_vector);
        for i in 0..rows - o {
            for j in 0.. cols - o {
                new.data[(o + i)*cols + j+ o] -= householder.beta * householder.vector[i] * householder.vector[j]
            }
        }
        println!("Left multiplication {:?}", new);
        a = tensor_mult(4, &new, &a);
        println!("Here's what the mult looks like check 0's {:?}", a);
        if  o < cols.min(rows) - 2 {
            new = create_identity_matrix(rows);
            let row_vector:Vec<f32> = a.data[(o*cols) + 1.. (o + 1) *cols ].to_vec();
            // println!("Row vector should be dim 3 and the top row {:?}", row_vector);
            householder = householder_params(&row_vector);

            for i in 0..rows - o - 1{
                for j in 0..cols - o - 1{
                    new.data[(o + i + 1)*cols + (j + o + 1) ] -= householder.beta * householder.vector[i] * householder.vector[j];
                }
            }
            // println!("This should be the matrix {:?}", new);
            a = tensor_mult(4, &a, &new);
            // println!("This is what a looks like {:?}", a);
        }
    }
    a
}

fn golub_kahan(mut a:NdArray) -> NdArray{
    let rows = a.dims[0];
    let cols = a.dims[1];
    let mut householder: HouseholderReflection = HouseholderReflection::new(0_f32, vec![0_f32;0]);

    let mut new:NdArray = create_identity_matrix(rows);
    let mut queue = vec![0_f32; rows  * cols ];

    for o in 0..cols.min(rows) - 1 {
        new = create_identity_matrix(rows);
        let mut column_vector = (o..rows).into_par_iter().map(|r| a.data[r*cols + o]).collect::<Vec<f32>>();
        householder = householder_params(&column_vector);
        for i in o..rows {
            for j in o..cols {
                for k in o..rows {
                    queue[i * cols + j] += householder.beta * householder.vector[i - o] * householder.vector[k - o] * a.data[ k*cols + j];
                }
            }
        }
        for i in o..cols {
            for j in o..rows {
                a.data[i*cols + j] -= queue[i*cols + j];
                queue[i*cols + j] = 0_f32;
            }
        }
        println!("matrix A {:?}", a);
        if  o < cols.min(rows) - 2 {
            // new = create_identity_matrix(rows);
            let row_vector:Vec<f32> = a.data[(o*cols) + 1.. (o + 1) *cols ].to_vec();
            // println!("Row vector should be dim 3 and the top row {:?}", row_vector);
            householder = householder_params(&row_vector);
            println!("Householder vector {:?}", householder.vector);
            for i in 0..rows{
                for j in o+1..cols{
                    // for k in o+1..rows{
                    for k in 0..rows{
                        // a[i, j] = Sum a[i, k] * h[k, j]
                        //
                        if (i==0) & (j==2) {
                            println!("a'[i:{}, j:{}] := a[i:{}, k:{}] * h[k:{}, j:{}]", i, j, i, k, k-o, j-o);
                            // println!("aik = {}, hkj = {}", a.data[i*cols + k], householder.vector[k - o - 1] * householder.vector[j - o - 1]);
                        }
                        // if j <= k {
                        if o < k {
                            if (i==0) & (j==2) {
                            println!("EXECUTING a'[i:{}, j:{}] := a[i:{}, k:{}] * h[k:{}, j:{}]", i, j, i, k, k-o, j-o);
                            }
                            queue[i*cols + j] += householder.beta * a.data[i *cols + k] * householder.vector[k - o - 1] * householder.vector[j - o - 1];
                        }
                    }
                }
            }
            println!("Check queue {:?}", queue);
            for i in 0..rows{
                // for j in o+1..cols{
                for j in 0..cols{
                    a.data[i * cols + j] -= queue[i *cols + j];
                    queue[i *cols + j] = 0_f32;
                }
            }
            println!("ensure queue is clear {:?}", queue);
            println!("Here's what the mult looks like check 0's {:?}", a);
        }
    }
    a
}




fn main() {
    let mut data:Vec<f32>;
    let mut dims:Vec<usize>;
    // data[0] = 1_f32;
    // data[1] = -1_f32;
    // data[2] = 4_f32;
    // data[3] = 1_f32;
    // {
    //     // Eigen values 2, -1
    //     let mut data = vec![0_f32; 4];
    //     let mut dims = vec![2; 2];
    //     data[0] = -1_f32;
    //     data[1] = 0_f32;
    //     data[2] = 5_f32;
    //     data[3] = 2_f32;
    // }
    {
        data = vec![0_f32; 9];
        dims = vec![3; 2];
        data[0] = 1_f32;
        data[1] = 2_f32;
        data[2] = 3_f32;
        data[3] = 3_f32;
        data[4] = 4_f32;
        data[5] = 5_f32;
        data[6] = 6_f32;
        data[7] = 7_f32;
        data[8] = 8_f32;
    }
    let x = NdArray::new(dims, data.clone());
    println!("x: {:?}", x);
    //
    let reference = explicit_golub_kahan(x.clone());
    println!("Reference {:?}", reference);

    let dev = golub_kahan(x.clone());
    println!("Implicit {:?}", dev);


    // let sym = symmetricize(x);
    // println!("Did it make symmetric? {:?}", sym);
    // let test = golub_kahan_lanczos(x.clone());
    // // println!("Test:\nU {:?}\nS {:?}\nV {:?}", test.0, test.1, test.2);
    // println!("Bidiagonal \nS {:?}", test.1);


    // let mut check = tensor_mult(2, &transpose(test.0), &test.1.clone());
    // check = tensor_mult(2, &check, &test.2.clone());
    // println!("Checking reconstruction {:?}", check);

    
    // let real_schur = real_schur_decomp(x.clone());
    // println!("real schur kernel {:?}", real_schur.kernel);
    
    // let sigma = bidiagonal_qr(test.1.clone());
    // println!("Bidiagonal QR {:?}", sigma);
    // println!("Expected eigen: 2, 1");
    
    // // From lapack white paper
    // let sigma = fast_bidiagonal_qr(fast_bidiagonal_qr(test.1.clone()));
    // println!("Fast Bidiagonal QR {:?}", sigma);
    // println!("Expected eigen: 2, 1");
    // // // println!("real schur rotation {:?}", real_schur.rotation);
    // //
    // let y = qr_decompose(x.clone());
    // println!("triangle {:?}", y.triangle);


    // let q = real_schur.rotation;
    // let q_star = transpose(q.clone());
    // println!("Schur rotation {:?}", q);
    // let q_orthogonality_check = tensor_mult(2, &q, &q_star);
    // println!("U orthogonality check {:?}", q_orthogonality_check);

    // let symmetric = make_symmetric(&real_schur.kernel);
    // println!("Symmetric values {:?}", symmetric);
    
    // // let eigen = givens_decomp(symmetric);
    // let eigen = eigen_square(y.triangle);
    // let eigen = jacobi_decomp(symmetric);
    // let eigen = givens_decomp(y.triangle);
    // let eigen = givens_decomp(x.clone());
    // println!("eigen values {:?}", eigen);
    
    // let eigen = givens_decomp(real_schur.kernel);
    // let eigen = jacobi_decomp(symmetric);
    // println!("eigen values {:?}", eigen.kernel);

}

