pub fn initialize_unit_vector(n:usize) -> Vec<f32> {
    vec![1_f32 / (n as f32).sqrt();n]
}

pub fn vector_diff(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn vector_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn vector_product(x: &[f32], y: &[f32]) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).collect()
}

pub fn scalar_product(lambda: f32, vector: &[f32]) -> Vec<f32> {
    vector.iter().map(|&vector| lambda * vector).collect()
}


pub fn dot_product(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn magnitude(x: &[f32]) -> f32 {
    x.iter()
        .zip(x.iter())
        .map(|(&x, &y)| x * y)
        .sum::<f32>()
        .sqrt()
}
