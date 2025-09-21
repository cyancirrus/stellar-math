use stellar::learning::knn::LshKNearestNeighbors;
use stellar::structure::ndarray::NdArray;
use stellar::decomposition::qr::qr_decompose;
use stellar::algebra::ndmethods::tensor_mult;

// #[cfg(target_arch = "x86_64")]
use rand::Rng;
use rand_distr::StandardNormal;
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;

// write article on how to do the (I - buu')(I- buu').. Q[i] -BQ[i]uu'
// also include the like Qx and QX like work
// motivate it by the eigenvecs

// move code into examples directory
// cargo run --example demo
    
const CONVERGENCE_CONDITION: f32 = 1e-6;

fn generate_random_matrix(dims:&[usize]) -> NdArray {
    debug_assert!(dims.len() > 0 && dims[0] > 0);
    let mut rng = rand::rng();

    let n = dims.iter().product();
    NdArray {
        dims:dims.to_vec(),
        data: (0..n).map(|_| rng.sample(StandardNormal)).collect(),
    }
}

fn sign_allign_difference(a:&NdArray, b:&NdArray) -> f32 {
    // distance :: SS (sign*a[ij] - b[ij])^2
    // sign := a'b
    debug_assert!(a.dims == b.dims);
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let mut error = 0_f32;
    for j in 0..cols {
        let mut dot = 0_f32;
        for i in 0..rows {
            dot += a.data[ i * cols +j ] * b.data[ i * cols + j ]
        }
        let sign = dot.signum();
        for i in 0..rows {
            let diff = sign * a.data[ i * cols + j] - b.data[ i * cols + j];
            error += diff * diff;
        }
    }
    error
}



fn random_eigenvector_decomp(matrix:NdArray) -> NdArray {
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
        .map(|_| (0..dim).map(|_| rng.random_range(-10.0..10.0) as f32).collect())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    
    for _ in 0..num_points {
        // pick a random cluster
        let c = &centers[rng.random_range(0..num_clusters)];
        // sample around center
        let point: Vec<f32> = c.iter()
            .map(|&v| v + normal.sample(&mut rng) as f32)
            .collect();
        data.push(point);
    }
    data
}

fn main() {
    let data = generate_clusters(100, 2, 3); // 100 points, 2D, 3 clusters
    // for p in &data {
    //     println!("{:?}", p);
    // }
    let mut knn = LshKNearestNeighbors::new(7, 2, 6); 
    knn.parse(data.clone());
    // for p in &data {
    //     println!("{:?}", p);
    // }
    let result = knn.knn(5, data[0].clone());
    println!("--------------");
    for p in &result {
        println!("{:?}", p);
    }
}
