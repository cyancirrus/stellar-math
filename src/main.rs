#![allow(unused)]
use stellar::structure::ndarray::NdArray;

struct HhMatrixRowMajor {}

///  Row major optimization form of LQ
struct AutumnDecomp {
    pub h: NdArray,
    card: usize,
    cols: usize,
    rows: usize,
}

impl HhMatrixRowMajor {
    fn params(v: &mut [f32], card: usize, rows: usize, p: usize) {
        let mut max_element = f32::NEG_INFINITY;
        let mut magnitude_squared = 0f32;
        for val in v.iter() {
            max_element = max_element.max(val.abs());
        }
        let inv_max_element = 1f32 / max_element;
        for val in v.into_iter() {
            *val *= inv_max_element;
            magnitude_squared += *val * *val;
        }
        let g = v[0].signum() * magnitude_squared.sqrt();
        let scale = v[0] + g;
        let inv_scale = 1f32 / scale;
        for val in v.into_iter().skip(1) {
            *val *= inv_scale;
        }
        v[0] = scale / g;
    }
}

impl AutumnDecomp {
    fn new(mut h: NdArray) -> Self {
        debug_assert!(h.dims[0] <= h.dims[1]);
        let (rows, cols) = (h.dims[0], h.dims[1]);
        let card = rows.min(cols);
        let mut buffer = vec![0f32; rows];
        let mut split_range= rows;
        for p in 0..card {
            split_range -= 1;
            let (projection, target) = h.data.split_at_mut((p + 1) * cols);
            let projection = &mut projection[p * cols + p..];
            HhMatrixRowMajor::params(projection, card, rows, p);
            // w' = v'H
            for i in 0..split_range {
                // target indexed at below active row
                let target_row = &target[i * cols..(i + 1) * cols];
                let mut wi = target_row[p];

                for j in p+1..cols {
                    wi += target_row[j] * projection[j - p];
                }
                buffer[i] = wi;
            }
            let tau = projection[0];
            // H -= T vw'
            // already indexed into rows below the householder
            for i in 0..split_range {
                let scalar = tau * buffer[i];
                let target_row = &mut target[i * cols..(i + 1) * cols];
                target_row[p] -= scalar;
                for j in p+1..cols {
                    target_row[j] -= scalar * projection[j - p];
                }
            }
        }
        Self {
            h,
            rows,
            cols,
            card,
        }
    }
}

fn main() {}
