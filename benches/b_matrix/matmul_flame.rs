use std::hint::black_box;
use stellar::algebra::mmethods::tensor_kernel_new;
use stellar::arch::SIMD_WIDTH;
use stellar::random::generation::generate_random_matrix;
use stellar::structure::ndarray::NdArray;

fn main() {
    // let i = 1024;
    // let k = 1024;
    // let j = 1024;

    // // Setup data once outside the loop
    // let x = generate_random_matrix(i, k);
    // let y = generate_random_matrix(k, j);
    // let mut target = vec![0.0f32; i * j];

    // println!("Starting profile loop...");

    // // Run for enough iterations to get a good sample (approx 5-10 seconds)
    // for _ in 0..500 {
    //     black_box(tensor_kernel_new(&x, &y, &mut target));
    // }

    // println!("Done.");
}
