use crate::structure::ndarray::NdArray;

fn jacobi_rotation(i:usize, j:usize, ndarray:&NdArray) -> NdArray {
    let cols = ndarray.dims[1];
    let mut jacobi = vec![0_f32; 4];
    let magnitude = ((ndarray.data[j*cols + j] + ndarray.data[i*cols + i]) / ndarray.data[j*cols + i]).powi(2);
    let s = 1_f32 / (magnitude + 1_f32).sqrt();
    let c = 1_f32 - s.powi(2);

    jacobi[0]=c;
    jacobi[1]=s;
    jacobi[2]=-s;
    jacobi[3]=c;
    NdArray::new(vec![2;2], jacobi)
}

