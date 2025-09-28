const TOLERANCE: f32 = 1e-3;

pub fn approx_vector_eq(a: &[f32], b: &[f32]) -> bool {
    let n = a.len();
    let mut error = 0_f32;
    for i in 0..n {
        error += (a[i] - b[i]).abs();
    }
    error / (n as f32).sqrt() < TOLERANCE
}

pub fn approx_scalar_eq(a:f32, b:f32) -> bool {
    (a-b).abs() < TOLERANCE
}
