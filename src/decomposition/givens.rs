use crate::structure::ndarray::NdArray;

const CONVERGENCE_CONDITION: f32 = 1e-6;

pub struct SingularValueDecomp {
    pub u: NdArray,
    pub s: NdArray,
    pub v: NdArray,
}
impl SingularValueDecomp {
    pub fn new(u: NdArray, s: NdArray, v: NdArray) -> Self {
        Self { s, u, v }
    }
}
pub fn full_givens_iteration(
    mut u: NdArray,
    mut s: NdArray,
    mut v: NdArray,
) -> SingularValueDecomp {
    // takes in bidiagonal and returns full SVD
    let m = s.dims[0];
    let n = s.dims[1];
    let k = m.min(n);
    // row-space, column-space
    let mut max_iteration = 1 << 8;
    // left work
    while offdiag_norm(&s) > CONVERGENCE_CONDITION && max_iteration > 0 {
        for i in 0..k - 1 {
            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[(i + 1) * n + i]);
            // below diagonal element
            apply_g_left(&mut s, i, i + 1, cosine, sine);
            apply_gt_right(&mut u, i, i + 1, cosine, sine);

            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[i * n + i + 1]);
            apply_gt_right(&mut s, i, i + 1, cosine, sine);
            apply_gt_right(&mut v, i, i + 1, cosine, sine);
        }
        max_iteration -= 1
    }
    SingularValueDecomp { u, s, v }
}
pub fn givens_iteration(mut s: NdArray) -> Vec<f32> {
    // takes in bidiagonal and returns full SVD
    let m = s.dims[0];
    let n = s.dims[1];
    let k = m.min(n);
    let mut singular = Vec::with_capacity(n);
    // row-space, column-space
    let mut max_iteration = 1 << 8;
    // left work
    while offdiag_norm(&s) > CONVERGENCE_CONDITION && max_iteration > 0 {
        for i in 0..k - 1 {
            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[(i + 1) * n + i]);
            // below diagonal element
            apply_g_left(&mut s, i, i + 1, cosine, sine);

            let (_, cosine, sine) =
                implicit_givens_rotation(s.data[i * n + i], s.data[i * n + i + 1]);
            apply_gt_right(&mut s, i, i + 1, cosine, sine);
        }
        max_iteration -= 1
    }
    for idx in 0..n {
        singular.push(s.data[idx * n + idx]);
    }
    singular
}
// m x n, m x m x n
fn offdiag_norm(s: &NdArray) -> f32 {
    let m = s.dims[0];
    let n = s.dims[1];
    let mut norm = 0.0;
    for i in 0..m.min(n) - 1 {
        // upper diagonal
        norm += s.data[i * n + i + 1].abs() + s.data[(i + 1) * n + i].abs();
    }
    norm
}
pub fn implicit_givens_rotation(a: f32, b: f32) -> (f32, f32, f32) {
    let t: f32;
    let tt: f32;
    let s: f32;
    let c: f32;
    let r: f32;

    if a == 0f32 {
        c = 0f32;
        s = 1f32;
        r = b;
    } else if b.abs() > a.abs() {
        t = a / b;
        tt = (1f32 + t * t).sqrt();
        s = 1f32 / tt;
        c = s * t;
        r = b * tt;
    } else {
        t = b / a;
        tt = (1f32 + t * t).sqrt();
        c = 1f32 / tt;
        s = c * t;
        r = a * tt;
    }
    // let r: f32 = (a.powi(2) + b.powi(2)).sqrt();
    (r, c, s)
}
pub fn apply_g_left(a: &mut NdArray, i: usize, j: usize, c: f32, s: f32) {
    // G * A
    // alpha, beta, gamma, delta,
    // c, s, -s, c
    let n = a.dims[1];
    let r1 = i * n;
    let r2 = j * n;
    for k in 0..n {
        // alpha a[i*,k] + beta a[j*, k];
        let i_replace = c * a.data[r1 + k] + s * a.data[r2 + k];
        // gamma a[i*,k] + delta a[j*, k];
        let j_replace = -s * a.data[r1 + k] + c * a.data[r2 + k];
        a.data[r1 + k] = i_replace;
        a.data[r2 + k] = j_replace;
    }
}
pub fn apply_g_right(a: &mut NdArray, i: usize, j: usize, c: f32, s: f32) {
    // A * G
    // alpha, beta, gamma, delta,
    // c, s, -s, c
    let (m, n) = (a.dims[0], a.dims[1]);
    let mut r = 0;
    for _ in 0..m {
        // alpha a[l,i*] + gamma a[l, j*];
        let i_replace = c * a.data[r + i] - s * a.data[r + j];
        // beta a[l,i*] + delta a[l, j*];
        let j_replace = s * a.data[r + i] + c * a.data[r + j];
        a.data[r + i] = i_replace;
        a.data[r + j] = j_replace;
        r += n;
    }
}
pub fn apply_gt_left(a: &mut NdArray, i: usize, j: usize, c: f32, s: f32) {
    // G' * A
    // transpose the negative sine
    // alpha, beta, gamma, delta,
    // c, -s, s, c
    apply_g_left(a, i, j, c, -s);
}
pub fn apply_gt_right(a: &mut NdArray, i: usize, j: usize, c: f32, s: f32) {
    // A * G'
    // alpha, beta, gamma, delta,
    // c, -s, s, c
    apply_g_right(a, i, j, c, -s);
}
