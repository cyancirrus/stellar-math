use crate::structure::ndarray::NdArray;

fn lu_factorization(x: &NdArray) -> (NdArray, NdArray) {
    let rows = x.dims[0];
    let cols = x.dims[1];
    assert_eq!(rows, cols, "currently LU is available only for square");
    let mut lower = vec![0_f32; x.data.len()];
    let mut upper = x.data.clone();

    for j in 0..rows {
        for i in 0..rows {
            for k in 0..rows {
                if j > i && k == 0 {
                    upper[j * cols + i] = 0_f32;
                } else if i == j && k == 0 {
                    lower[i * cols + j] = 1_f32;
                } else if i > j {
                    if k == 0 {
                        lower[i * cols + j] = -upper[i * cols + j] / upper[j * cols + j];
                        upper[i * cols + j] = 0_f32;
                    } else {
                        upper[i * cols + k] += lower[i * cols + j] * upper[j * cols + k];
                    }
                }
            }
        }
    }
    (
        NdArray::new(x.dims.clone(), lower),
        NdArray::new(x.dims.clone(), upper),
    )
}
