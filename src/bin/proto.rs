// #[cfg(target_arch = "x86_64")]

// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR


// move code into examples directory
// cargo run --example demo
use stellar::decomposition::qr::{qr_decompose};
use stellar::decomposition::lu::{lu_decompose};
use stellar::structure::ndarray::NdArray;
use stellar::random::generation::{generate_random_matrix, generate_random_symetric};

const CONVERGENCE_CONDITION: f32 = 1e-4;

fn rayleigh_inverse_iteration(mut a: NdArray) -> NdArray {
    // (A - Iu)y = x;
    // x' = Q from QR(y)
    // let M := (A-Iu)
    // LU(M) -> solve => y
    // u' := rayleigh quotient of x
    debug_assert!(a.dims.len() == 2);
    debug_assert!(a.dims[0] == a.dims[1]);
    let n = a.dims[0];
    let mut x_cur = generate_random_matrix(n, n);
    let mut u: NdArray;
    let mut m: NdArray;
    let mut error = 1_f32;
    while CONVERGENCE_CONDITION < error {
        let x_pre = x_cur.clone();
        u = estimate_eigenvalues(&mut a, &x_cur);
        m = determine_m(&a, &u, &x_cur);
        let lu = lu_decompose(m);
        lu.solve_inplace(&mut x_cur);
        println!("before qr");
        let qr = qr_decompose(x_cur);
        println!("after qr");
        x_cur = qr.projection_matrix();
        error = frobenius_diff_norm(&x_cur, &x_pre);
        println!("error: {error:?}");
    }
    x_cur
}

fn determine_m(a:&NdArray,  u:&NdArray, x:&NdArray) -> NdArray {
    // M := A - XUX' == A - XX'UXX'
    let (n, k) = (a.dims[1], u.dims[0]);
    let mut m = a.clone();
    
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                for q in 0..k {
                    sum += x.data[i * k + p] * u.data[p * k + q] * x.data[j * k + q];
                }
            }
            m.data[i * n + j] = a.data[i * n + j] - sum;
        }
    }
    m
}
    
fn frobenius_diff_norm(a: &NdArray, b: &NdArray) -> f32 {
    // distance :: SS (sign*a[ij] - b[ij])^2
    // sign := a'b
    debug_assert!(a.dims == b.dims);
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let mut error = 0_f32;
    for j in 0..cols {
        for i in 0..rows {
            let diff = a.data[i * cols + j] - b.data[i * cols + j];
            error += diff * diff;
        }
    }
    (error / (rows * cols) as f32).sqrt()
}

fn estimate_eigenvalues(a: &mut NdArray, x:&NdArray) -> NdArray {
    // estimated via rayleigh quotient
    // x'Ax/x'x
    // k for eventual subsetting
    debug_assert_eq!(a.dims[0], a.dims[1]); 
    let (n, k) = (a.dims[0], x.dims[1]);
    let mut ax = vec![0_f32; n * k];
    let mut xt_a_x = vec![0_f32; k*k];
    // A ~ n,n
    // Ax ~ n, k
    // x'n, k ~ k,k
    for i in 0..n {
        for j in 0..k {
            for m in 0..n {
                ax[i * k + j] += a.data[i * n + m] * x.data[ m * k + j];
            }
        }
    }
    for i in 0..k {
        for j in 0..k {
            for m in 0..n {
                // x is transposed, either x or ax needs to be transposed x is nxn
                xt_a_x[i * k + j] += x.data[m * k + i] * ax[m * k + j];
            }
        }
    }
    NdArray {
        dims: vec![k,k],
        data: xt_a_x,
    }
}

fn test_random_eigenvectors() {
    let n = 3;

    // Step 1: create a random symmetric matrix
    let matrix = generate_random_symetric(n);

    // Step 2: run your eigenvector decomposition
    let eigenvecs = rayleigh_inverse_iteration(matrix.clone());

    // Step 3: check orthonormality Q^T Q ~ I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += eigenvecs.data[k * n + i] * eigenvecs.data[k * n + j];
            }
            if i == j {
                assert!((dot - 1.0).abs() < 1e-3, "Column {} not normalized", i);
            } else {
                assert!(dot.abs() < 1e-3, "Columns {} and {} not orthogonal", i, j);
            }
        }
    }

    // Step 4: check eigenvector property A v ~ lambda v
    for col in 0..n {
        let mut a_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                a_v[i] += matrix.data[i * n + j] * eigenvecs.data[j * n + col];
            }
        }

        // estimate lambda as ratio of norms
        let mut lambda_est = 0.0;
        let mut norm_v = 0.0;
        for i in 0..n {
            lambda_est += a_v[i] * eigenvecs.data[i * n + col];
            norm_v += eigenvecs.data[i * n + col].powi(2);
        }
        lambda_est /= norm_v;

        // check that A v ~ lambda v
        for i in 0..n {
            let diff = (a_v[i] - lambda_est * eigenvecs.data[i * n + col]).abs();
            assert!(
                diff < 1e-3,
                "Eigenvector column {} failed A v ~ lambda v, diff={}",
                col,
                diff
            );
        }
    }

    println!("All tests passed!");
}


fn main() {
    // it's in proto.bu
    test_random_eigenvectors();
}
