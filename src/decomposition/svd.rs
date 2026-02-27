use crate::algebra::ndmethods::create_identity_matrix;
use crate::decomposition::householder::{
    HouseholderReflection, householder_inplace, householder_params,
};
use crate::structure::ndarray::NdArray;

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
        // let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
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
    let card = rows.min(cols);
    a.resize(card, card);
    a
}

pub fn full_golub_kahan_old(mut a: NdArray) -> (NdArray, NdArray, NdArray) {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut u = create_identity_matrix(rows);
    let mut v = create_identity_matrix(cols);
    let mut proj: HouseholderReflection;
    let mut sum: f32;
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
            for k in o..cols {
                sum += u.data[i * rows + k] * proj.vector[k - o];
            }
            sum *= proj.beta;
            for k in o..cols {
                u.data[i * rows + k] -= sum * proj.vector[k - o];
            }
        }
        // stop one early for columns because a[m,n-1] can be non-zero
        if o + 1 == card {
            break;
        }
        // let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
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
            sum = 0f32;
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

pub fn full_golub_kahan(mut a: NdArray) -> (NdArray, NdArray, NdArray) {
    // singular vlaues are up to sign convention
    // ie if l[k] < 0 => sign flip u[k]
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let card = rows.min(cols) - (rows <= cols) as usize;
    let mut u = create_identity_matrix(rows);
    let mut v = create_identity_matrix(cols);
    let mut house = vec![0f32; rows.max(cols)];
    let mut buffer = vec![0f32; rows.max(cols)];
    let mut sum: f32;
    let mut beta: f32;
    for o in 0..card {
        for r in o..rows {
            house[r] = a.data[r * cols + o];
        }
        beta = householder_inplace(&mut house);
        // (I - bvv')A => w := bv'A
        //  A -= vw'
        // w' = u'X
        buffer[..cols].fill(0f32);
        for i in o..rows {
            let target_row = &a.data[i * cols..(i + 1) * cols];
            let v_i = house[i - o];
            for j in 0..cols {
                buffer[j] += v_i * target_row[j];
            }
        }
        // X -= B uw'
        // for i in p..self.cols {
        for i in o..rows {
            let scalar = beta * house[i - o];
            let target_row = &mut a.data[i * cols..(i + 1) * cols];
            for j in 0..cols {
                target_row[j] -= scalar * buffer[j];
            }
        }
        // U(I - bvv')' = U(I - bvv')
        for i in 0..rows {
            sum = 0f32;
            for k in o..cols {
                sum += u.data[i * rows + k] * house[k - o];
            }
            sum *= beta;
            for k in o..cols {
                u.data[i * rows + k] -= sum * house[k - o];
            }
        }
        // stop one early for columns because a[m,n-1] can be non-zero
        if o + 1 == card {
            break;
        }
        // let row_vector = a.data[(o * cols) + 1 + o..(o + 1) * cols].to_vec();
        for r in o + 1..cols {
            house[r] = a.data[o * cols + r];
        }
        beta = householder_inplace(&mut house[o + 1..cols]);
        // A(I - bvv') => w = b * Av
        // A -= wv'
        for i in o..rows {
            sum = 0f32;
            for j in o + 1..cols {
                sum += a.data[i * cols + j] * house[j - o - 1];
            }
            sum *= beta;
            for j in o + 1..cols {
                a.data[i * cols + j] -= sum * house[j - o - 1];
            }
        }
        // (I - bvv')'V' ~ V(I- bvv')
        // v ~ (r1 r2 r3 r4)
        for j in 0..cols {
            // inner product of v[i..] * b;
            sum = 0f32;
            for k in o + 1..cols {
                sum += v.data[j * cols + k] * house[k - o - 1];
            }
            sum *= beta;
            for k in o + 1..cols {
                v.data[j * cols + k] -= sum * house[k - o - 1];
            }
        }
    }
    (u, a, v)
}
