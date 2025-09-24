use crate::algebra::ndmethods::create_identity_matrix;
use crate::decomposition::givens::givens_iteration;
use crate::decomposition::qr::qr_decompose;
use crate::decomposition::schur::real_schur;
use crate::decomposition::svd::golub_kahan_explicit;
use crate::structure::ndarray::NdArray;

const TOLERANCE_CONDITION: f32 = 1e-6;

struct LU {
    lower: NdArray,
    upper: NdArray,
}

fn lower_upper(mut upper: NdArray) -> LU {
    // A[j, *] = c *A[i, *]
    // => c = A[i,j] / A[j,j]
    let rows = upper.dims[0];
    let cols = upper.dims[1];
    debug_assert_eq!(rows, cols);
    let mut lower = create_identity_matrix(rows);

    for i in 0..cols {
        for j in i..rows {
            if upper.data[j * cols + i].abs() < TOLERANCE_CONDITION {
                continue;
            }
            let c = upper.data[j * cols + i] / upper.data[i * cols + i];
            lower.data[j * cols + i] = c;
            for k in i..cols {
                upper.data[j * cols + k] -= c * upper.data[i * cols + k];
            }
        }
    }
    LU { lower, upper }
}
