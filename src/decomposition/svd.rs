use crate::algebra::ndmethods::{
    create_identity_matrix, matrix_mult, resize_cols, resize_rows, tensor_mult, transpose,
};
use crate::algebra::vector::{initialize_unit_vector, magnitude};
use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::ParallelIterator;
use rayon::prelude::*;

// TODO: store the bidiagonal

// halko-trop
pub fn golub_kahan(mut a: NdArray) -> NdArray {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut proj: HouseholderReflection;
    let mut sum;
    for o in 0..card {
        proj = householder_params(
            // column vector
            (o..rows)
                .into_iter()
                .map(|r| a.data[r * cols + o])
                .collect(),
        );
        // (I - bvv')A => w := bv'A
        //  A -= vw'
        for j in o..cols {
            sum = 0f32;
            for i in o..rows {
                sum += proj.vector[i - o] * a.data[i * cols + j];
            }
            sum *= proj.beta;
            for i in o..rows {
                a.data[i * cols + j] -= proj.vector[i - o] * sum;
            }
        }
        // stop one early for columns because a[m,n-1] can be non-zero
        if o + 1 == card {
            break;
        }
        let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
        proj = householder_params(
            // row vector
            a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec(),
        );
        // A(I - bvv') => w = b * Av
        // A -= wv'
        for i in o..rows {
            sum = 0f32;
            for j in o + 1..cols {
                sum += a.data[i * cols + j] * proj.vector[j - o - 1];
            }
            sum *= proj.beta;
            for j in o + 1..cols {
                a.data[i * cols + j] -= sum * proj.vector[j - o - 1];
            }
        }
    }
    let dims = rows.min(cols);
    resize_rows(dims, &mut a);
    resize_cols(dims, &mut a);
    a
}

pub fn full_golub_kahan(mut a: NdArray) -> (NdArray, NdArray, NdArray) {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut u = create_identity_matrix(rows);
    let mut v = create_identity_matrix(cols);
    let mut proj: HouseholderReflection;
    let mut sum:f32;
    for o in 0..card {
        proj = householder_params(
            // column vector
            (o..rows).map(|r| a.data[r * cols + o]).collect(),
        );
        // (I - bvv')A => w := bv'A
        //  A -= vw'
        for j in o..cols {
            sum = 0f32;
            for i in o..rows {
                sum += proj.vector[i - o] * a.data[i * cols + j];
            }
            sum *= proj.beta;
            for i in o..rows {
                a.data[i * cols + j] -= proj.vector[i - o] * sum;
            }
        }
        // U(I - bvv')' = U(I - bvv')
        for i in 0..rows {
            sum = 0f32;
            for k in o..rows {
                sum += u.data[i * rows + k] * proj.vector[k - o];
            }
            sum *= proj.beta;
            for k in o..rows {
                u.data[i * rows + k] -= sum * proj.vector[k - o];
            }
        }
        // stop one early for columns because a[m,n-1] can be non-zero
        if o + 1 == card {
            break;
        }
        let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
        proj = householder_params(
            // row vector
            a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec(),
        );
        // A(I - bvv') => w = b * Av
        // A -= wv'
        for i in o..rows {
            sum = 0f32;
            for j in o + 1..cols {
                sum += a.data[i * cols + j] * proj.vector[j - o - 1];
            }
            sum *= proj.beta;
            for j in o + 1..cols {
                a.data[i * cols + j] -= sum * proj.vector[j - o - 1];
            }
        }
        // (I - bvv')'V' ~ V(I- bvv')
        // v ~ (r1 r2 r3 r4)
        for j in 0..cols {
            // inner product of v[i..] * b;
            sum = 0_f32;
            for k in o + 1..cols {
                sum += v.data[j * cols + k] * proj.vector[k - o - 1];
            }
            sum *= proj.beta;
            for k in o + 1..cols {
                v.data[j * cols + k] -= sum * proj.vector[k - o - 1];
            }
        }
    }
    (u, a, v)
}
