use crate::structure::ndarray::NdArray;

use crate::arch::SIMD_WIDTH;
const MC: usize = 40;
const PC: usize = 160;
const NC: usize = 120;

pub fn increment(basis: &mut [f32], data: &[f32], m: usize, n: usize, s_b: usize, s_d: usize) {
    let mut boffset = 0;
    let mut doffset = 0;
    for _ in 0..m {
        for j in 0..n {
            basis[boffset + j] += data[doffset + j];
        }
        boffset += s_b;
        doffset += s_d;
    }
}
pub fn filter_upper_trapezoid(a: &mut NdArray) {
    let (m, n) = (a.dims[0], a.dims[1]);
    let data = &mut a.data;
    let d_sub = m.saturating_sub(n);
    for i in 0..m {
        for j in 0..n {
            if j + d_sub >= i {
                break;
            }
            data[i * n + j] = 0f32;
        }
    }
}
/// * * * * * * * 0 0 0
/// * * * * * * * * 0 0
/// * * * * * * * * * 0
/// * * * * * * * * * *
pub fn filter_lower_trapezoid(a: &mut NdArray) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let t = cols.min(rows);
    let s = rows.saturating_sub(cols);
    // don't remove from last row
    for i in 1..t {
        for j in 0..i {
            d[(rows - i - s) * cols - j - 1] = 0f32;
        }
    }
}
pub fn test_data() -> Vec<(usize, usize, usize)> {
    vec![
        // (9, 16, 9),
        // (32, 32, 32),
        // (1, 1, 1),
        // (16, 16, 16),
        (8, 9, 8),
        (3, 9, 1),
        (6, 4, 8),
        (9, 16, 8),
        (8, 8, 9),
        (2, 9, 1),
        (2, 2, 1),
        (2, 9, 1),
        (2, 10, 1),
        (1, 9, 1),
        (4, 8, 1),
        (1, 2, 1),
        (1, 1, 1),
        (8, 1, 1),
        (1, 8, 1),
        (1, 1, 8),
        (6, 4, 8),
        (6, 8, 4),
        (8, 4, 6),
        (4, 8, 6),
        (4, 6, 8),
        (8, 6, 4),
        (8, 8, 8),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH + 1, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH + 1, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH + 1),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH - 1, SIMD_WIDTH, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH - 1, SIMD_WIDTH),
        (SIMD_WIDTH, SIMD_WIDTH, SIMD_WIDTH - 1),
        (MC + 1, PC, NC + 1),
        (MC + 1, PC, NC - 1),
        (MC + 1, PC, NC),
        (MC - 1, PC, NC),
        (MC, PC + 1, NC),
        (MC, PC - 1, NC),
        (MC, PC, NC),
        // (256, 256, 256),
        // (256, 1024, 512),
        // (512, 512, 512),
        // (1024, 64, 1024),
    ]
}
