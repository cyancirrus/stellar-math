use crate::decomposition::wy::constants::EPSILON;
/// copies memory from b into a
pub fn import_slice(target: &mut [f32], data: &[f32]) {
    target[..data.len()].copy_from_slice(data);
    // target[data.len()..].fill(0f32);
    // target[data.len()..];
}
/// params
///
/// takes in a slice, where we find the rotation vector
/// in order when multiplied by the original matrix returns
/// a zero'd matrix
pub fn params(v: &mut [f32]) -> f32 {
    let mut max_element = 0f32;
    for val in v.iter() {
        let v = val.abs();
        if v > max_element {
            max_element = v
        };
    }
    if max_element.abs() < EPSILON {
        return max_element;
    }
    let mut magnitude_squared = 0f32;
    let inv_max_element = 1f32 / max_element;
    for val in v.iter_mut() {
        *val *= inv_max_element;
        magnitude_squared += *val * *val;
    }
    let g = v[0].signum() * magnitude_squared.sqrt();
    let scale = v[0] + g;
    let inv_scale = 1f32 / scale;
    for val in v[1..].iter_mut() {
        *val *= inv_scale;
    }
    v[0] = -g * max_element;
    scale / g
}
