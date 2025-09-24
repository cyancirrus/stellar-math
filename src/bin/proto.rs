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

// TODO: Respond to the email asking to speak from founder


// #[cfg(target_arch = "x86_64")]
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Normal;
use rand_distr::StandardNormal;

// write article on how to do the (I - buu')(I- buu').. Q[i] -BQ[i]uu'
// also include the like Qx and QX like work
// motivate it by the eigenvecs

// move code into examples directory
// cargo run --example demo

const CONVERGENCE_CONDITION: f32 = 1e-6;

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
    while CONVERGENCE_CONDITION < error {
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

fn main() {
    // {
    // Eigen values 2, -1
    let mut data = vec![0_f32; 4];
    let dims = vec![2; 2];
    data[0] = -1_f32;
    data[1] = 0_f32;
    data[2] = 5_f32;
    data[3] = 2_f32;
    // }
    // {
    //     data = vec![0_f32; 9];
    //     dims = vec![3; 2];
    //     data[0] = 1_f32;
    //     data[1] = 2_f32;
    //     data[2] = 3_f32;
    //     data[3] = 3_f32;
    //     data[4] = 4_f32;
    //     data[5] = 5_f32;
    //     data[6] = 6_f32;
    //     data[7] = 7_f32;
    //     data[8] = 8_f32;
    // }
    let x = NdArray::new(dims, data.clone());
    println!("x: {:?}", x);
    //
    let reference = golub_kahan_explicit(x.clone());
    println!("Reference {:?}", reference);

    let y = qr_decompose(x.clone());
    println!("triangle {:?}", y.triangle);

    let real_schur = real_schur(x.clone());
    // eigenvalues
    println!("real schur kernel {:?}", real_schur.kernel);

    let svd = givens_iteration(reference);
    println!(
        "svd u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",
        svd.u, svd.s, svd.v
    );

    let evector = retrieve_eigen(real_schur.kernel.data[3], x.clone());
    println!("eigen vec {evector:?}");
}

// fn main() {
//     let data = generate_clusters(100, 2, 3); // 100 points, 2D, 3 clusters
//     // for p in &data {
//     //     println!("{:?}", p);
//     // }
//     let mut knn = LshKNearestNeighbors::new(7, 2, 6);
//     knn.parse(data.clone());
//     // for p in &data {
//     //     println!("{:?}", p);
//     // }
//     let result = knn.knn(5, data[0].clone());
//     println!("--------------");
//     for p in &result {
//         println!("{:?}", p);
//     }
// }
