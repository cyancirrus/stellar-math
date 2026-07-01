// Need to migrate the approx equal to utilizing condition numbers

const TOLERANCE: f32 = 1e-2;
// epsilon for f32 ~= 1e-7, but conditino number is an estimate;
const EPSILON: f32 = 1e-6;

pub fn approx_scalar_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < TOLERANCE
}
pub fn approx_vector_tol_eq(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    let n = a.len();
    let mut error = 0f32;
    for i in 0..n {
        if a[i].is_nan() || b[i].is_nan() {
            return false;
        }
        error += (a[i] - b[i]).abs();
    }
    error / (n as f32).sqrt() < tolerance
}
pub fn approx_vector_eq(a: &[f32], b: &[f32]) -> bool {
    approx_vector_tol_eq(a, b, TOLERANCE)
}
pub fn approx_condition_eq(a: &[f32], b: &[f32], k: &f32) -> bool {
    let mut diff_norm = 0f32;
    let mut target_norm = 0f32;
    let n = a.len();
    for i in 0..n {
        diff_norm += (a[i] - b[i]).abs();
        target_norm += b[i].abs();
    }
    (diff_norm / target_norm.max(f32::MIN_POSITIVE)) <= EPSILON * k
}
pub fn approx_stride_eq(
    actual: &[f32],
    expect: &[f32],
    m: usize,
    n: usize,
    s_a: usize,
    s_e: usize,
) -> bool {
    let mut error = 0f32;
    let len = m * n;
    for i in 0..m {
        for j in 0..n {
            let d = actual[i * s_a + j] - expect[i * s_e + j];
            if d.is_nan() {
                return false;
            }
            error += d.abs();
        }
    }
    error / (len as f32).sqrt() < TOLERANCE
}
