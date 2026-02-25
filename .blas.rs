use std::time::{Instant, Duration};
use ndarray::{Array2, ShapeBuilder, s};
use ndarray_linalg::{SVD, QR}; 
use std::hint::black_box;
use rand::Rng;

// Assuming these exist in your stellar crate as per your previous snippet
use stellar::random::generation::generate_random_matrix;

fn main() {
    let n = 1000;
    let k = 20;           // Target rank
    let p = 10;           // Oversampling for stability
    let iterations = 20;

    // 1. Generate data using your stellar function
    let matrix_data = generate_random_matrix(n, n);
    
    // 2. Load into ndarray with Fortran (.f()) layout for Accelerate compatibility
    let a: Array2<f32> = Array2::from_shape_vec((n, n).f(), matrix_data.data)
        .expect("Failed to create ndarray from matrix data");

    println!("--- Rank-K Randomized SVD with Power Iterations ---");
    println!("Matrix: {}x{}, Target Rank: {}, Oversampling: {}", n, n, k, p);

    let mut total_duration = Duration::ZERO;

    for i in 0..iterations {
        let start = Instant::now();

        // --- STEP 1: RANDOM SKETCH ---
        let mut rng = rand::rng();
        let mut omega = Array2::zeros((n, k + p).f());
        omega.map_inplace(|x| *x = rng.random::<f32>());

        // --- STEP 2: POWER ITERATION (AA')A Omega ---
        // This improves accuracy significantly for matrices with slow decay
        let y_0 = a.dot(&omega);
        let y_1 = a.t().dot(&y_0);
        let y = a.dot(&y_1);

        // --- STEP 3: ORTHOGONAL BASIS (QR) ---
        let (q, _r) = y.qr().expect("QR decomposition failed");
        let q_thin = q.slice(s![.., ..k + p]);

        // --- STEP 4: PROJECT DOWN ---
        // B = Q' * A  -> Resulting size is (k+p) x n
        let b = q_thin.t().dot(&a);

        // --- STEP 5: SMALL SVD ---
        // We perform SVD on the tiny B matrix (e.g., 30x1000)
        let (u_small_opt, s_vals, _vt) = b.svd(true, false).expect("SVD failed");
        let u_small = u_small_opt.unwrap();

        // --- STEP 6: PROJECT BACK UP ---
        // U = Q * U_small
        let u = q_thin.dot(&u_small);

        let elapsed = start.elapsed();
        
        // Use black_box so the compiler doesn't "optimize away" our hard work
        black_box((u, s_vals));

        println!("Iteration {}: {:?}", i + 1, elapsed);
        total_duration += elapsed;
    }

    println!("--------------------------------------------------");
    println!("Average Rank-K SVD time: {:?}", total_duration / iterations);
}
