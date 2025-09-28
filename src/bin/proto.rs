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


fn main() {
}
