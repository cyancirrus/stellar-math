// Need to migrate the approx equal to utilizing condition numbers

const TOLERANCE: f32 = 1e-2;
// epsilon for f32 ~= 1e-7, but conditino number is an estimate;
const EPSILON: f32 = 1e-6;

pub fn approx_vector_eq(a: &[f32], b: &[f32]) -> bool {
    let n = a.len();
    let mut error = 0_f32;
    for i in 0..n {
        error += (a[i] - b[i]).abs();
    }
    error / (n as f32).sqrt() < TOLERANCE
}

pub fn approx_scalar_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < TOLERANCE
}

pub fn approx_condition_eq(a: &[f32], b: &[f32], k: &f32) -> bool {
    let mut diff_norm = 0_f32;
    let mut target_norm = 0_f32;
    let n = a.len();
    for i in 0..n {
        diff_norm += (a[i] - b[i]).abs();
        target_norm += b[i].abs();
    }
    (diff_norm / target_norm.max(f32::MIN_POSITIVE)) <= EPSILON * k
}
