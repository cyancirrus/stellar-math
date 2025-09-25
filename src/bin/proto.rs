use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::tensor_mult;
use stellar::decomposition::givens::givens_iteration;
use stellar::decomposition::householder::HouseholderReflection;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::learning::knn::LshKNearestNeighbors;
use stellar::solver::eigenvector::retrieve_eigen;
use stellar::structure::ndarray::NdArray;

// #[cfg(target_arch = "x86_64")]
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Normal;
use rand_distr::StandardNormal;

// move code into examples directory
// cargo run --example demo

// const CONVERGENCE_CONDITION: f32 = 1e-6;
const CONVERGENCE_CONDITION: f32 = 1e-3;

fn generate_random_matrix(dims: &[usize]) -> NdArray {
    debug_assert!(dims.len() > 0 && dims[0] > 0);
    let mut rng = rand::rng();

    let n = dims.iter().product();
    NdArray {
        dims: dims.to_vec(),
        data: (0..n).map(|_| rng.sample(StandardNormal)).collect(),
    }
}

fn sign_allign_difference(a: &NdArray, b: &NdArray) -> f32 {
    // distance :: SS (sign*a[ij] - b[ij])^2
    // sign := a'b
    debug_assert!(a.dims == b.dims);
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let mut error = 0_f32;
    for j in 0..cols {
        let mut dot = 0_f32;
        for i in 0..rows {
            dot += a.data[i * cols + j] * b.data[i * cols + j]
        }
        let sign = dot.signum();
        for i in 0..rows {
            let diff = sign * a.data[i * cols + j] - b.data[i * cols + j];
            error += diff * diff;
        }
    }
    error
}

fn random_eigenvector_decomp(matrix: NdArray) -> NdArray {
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let mut eigen = generate_random_matrix(&matrix.dims);
    let mut error = 1_f32;
    for i in 0..4 {
    // while CONVERGENCE_CONDITION < error {
        let next = tensor_mult(4, &eigen, &matrix);
        let qr = qr_decompose(next);
        let projection = qr.projection_matrix();
        error = sign_allign_difference(&projection, &eigen);
        eigen = projection;
    }
    eigen
}

fn generate_clusters(num_points: usize, dim: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let mut data = Vec::new();

    // random cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            (0..dim)
                .map(|_| rng.random_range(-10.0..10.0) as f32)
                .collect()
        })
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();

    for _ in 0..num_points {
        // pick a random cluster
        let c = &centers[rng.random_range(0..num_clusters)];
        // sample around center
        let point: Vec<f32> = c
            .iter()
            .map(|&v| v + normal.sample(&mut rng) as f32)
            .collect();
        data.push(point);
    }
    data
}

fn quick_outer_product(reflection: HouseholderReflection) -> NdArray {
    let length = reflection.vector.len();

    // let data = vec![0;length * length];
    let mut ndarray = create_identity_matrix(length);
    for i in 0..length {
        for j in 0..length {
            ndarray.data[i * length + j] -=
                reflection.beta * reflection.vector[i] * reflection.vector[j];
        }
    }
    ndarray
}

fn test_random_eigenvectors() {
    let n = 3;

    // Step 1: create a random symmetric matrix
    let mut rng = rand::rng();
    let mut data = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in i..n {
            let val = rng.sample(StandardNormal);
            data[i * n + j] = val;
            data[j * n + i] = val; // symmetric
        }
    }
    let matrix = NdArray {
        dims: vec![n, n],
        data,
    };

    // Step 2: run your eigenvector decomposition
    let eigenvecs = random_eigenvector_decomp(matrix.clone());

    // Step 3: check orthonormality Q^T Q ~ I
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0;
            for k in 0..n {
                dot += eigenvecs.data[k * n + i] * eigenvecs.data[k * n + j];
            }
            if i == j {
                assert!((dot - 1.0).abs() < 1e-4, "Column {} not normalized", i);
            } else {
                assert!(dot.abs() < 1e-4, "Columns {} and {} not orthogonal", i, j);
            }
        }
    }

    // Step 4: check eigenvector property A v ~ lambda v
    for col in 0..n {
        let mut Av = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                Av[i] += matrix.data[i * n + j] * eigenvecs.data[j * n + col];
            }
        }

        // estimate lambda as ratio of norms
        let mut lambda_est = 0.0;
        let mut norm_v = 0.0;
        for i in 0..n {
            lambda_est += Av[i] * eigenvecs.data[i * n + col];
            norm_v += eigenvecs.data[i * n + col].powi(2);
        }
        lambda_est /= norm_v;

        // check that A v ~ lambda v
        for i in 0..n {
            let diff = (Av[i] - lambda_est * eigenvecs.data[i * n + col]).abs();
            assert!(diff < 1e-3, "Eigenvector column {} failed A v ~ lambda v, diff={}", col, diff);
        }
    }

    println!("All tests passed!");
}


fn main() {
    test_random_eigenvectors();
    // let data = generate_clusters(100, 2, 3); // 100 points, 2D, 3 clusters
    // // for p in &data {
    // //     println!("{:?}", p);
    // // }
    // let mut knn = LshKNearestNeighbors::new(7, 2, 6);
    // knn.parse(data.clone());
    // // for p in &data {
    // //     println!("{:?}", p);
    // // }
    // let result = knn.knn(5, data[0].clone());
    // println!("--------------");
    // for p in &result {
    //     println!("{:?}", p);
    // }
}
