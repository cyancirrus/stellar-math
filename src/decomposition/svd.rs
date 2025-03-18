use crate::structure::ndarray::NdArray;
use crate::algebra::ndmethods::{transpose, tensor_mult};
use crate::algebra::vector::{
    initialize_unit_vector,
    magnitude,
};

#[allow(non_snake_case)]
pub fn golub_kahan_lanczos(a:NdArray) -> (NdArray, NdArray, NdArray) {
    let n = a.dims[0];
    let mut U = NdArray::new(a.dims.clone(), vec![0_f32; n * n]);
    let mut V = NdArray::new(a.dims.clone(), vec![0_f32; n * n]);
    let mut beta = 0_f32;
    let mut alpha ;
    let mut v = initialize_unit_vector(n);
    let mut u= vec![0_f32; n];
    for i in 0..n  {
        for k in 0..n {
            V.data[i * n  + k] = v[k];
        }
        u.iter_mut().for_each(|val| *val *= -beta);
        {
            for j in 0..n {
                for k in 0..n {
                    u[j] += a.data[j * n + k] * v[k];
                }
            }
        }
        alpha = magnitude(&u);
        u.iter_mut().for_each(|val| *val /= alpha);
        for j in 0..n {
            U.data[i * n + j] = u[j];
        }
        {
            v.iter_mut().for_each(|val| *val *= -alpha);
            for j in 0..n {
                for k in 0..n {
                    v[j] += a.data[k * n + j] * u[k];
                }
            }
        }
        beta = magnitude(&v);
        v.iter_mut().for_each(|val| *val /= beta);
    }
    let mut S = tensor_mult(2, &U.clone(), &a);
    S = tensor_mult(3, &S, &transpose(V.clone()));
    println!("Check U {:?}", U);
    let mut O_check = tensor_mult(2, &U, &transpose(U.clone()));
    println!("Check U is orthogonal {:?}", O_check);
    O_check = tensor_mult(2, &V, &transpose(V.clone()));
    println!("Check V is orthogonal {:?}", O_check);

    (U, S, V)
}
