use crate::structure::ndsignal::{Complex, NdSignal};
use std::f32::consts::PI;

pub fn complex_tensor_mult(a:NdSignal, b:NdSignal) -> NdSignal {
    let mut dims = vec![0; 2];
    let mut data = vec![Complex::new(0_f32, 0_f32); a.dims[0] * b.dims[1]];
    dims[0] = a.dims[0];
    dims[1] = b.dims[1];

    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..a.dims[1] {
                data[i*dims[1] + j] += a.data[i * a.dims[1] + k] * b.data[k * b.dims[1] + j]
            }
        }
    }
    NdSignal::new(dims, data)
}

pub fn create_dct_array(n:usize) -> NdSignal {
    let dims = vec![n; 2];
    let mut data: Vec<Complex> = Vec::with_capacity(n * n);
    let mut row:usize;
    let mut col:usize;
    let mut val:Complex;
    let mut phase:f32;
    // disableling for fft debugging
    // let scalar = 1_f32 / (n as f32).sqrt();
    let scalar = 1_f32 ;
    for i in 0..n {
        for j in 0..n {
            phase = -2.0 * PI * (i as f32) * (j as f32) / (n as f32);
            val = Complex::new(
                scalar * phase.cos(),
                scalar * -phase.sin(),
            );
            data.push(val)
        }
    }
    NdSignal::new(dims, data)
}
