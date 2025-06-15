#[cfg(target_arch = "x86_64")]
use std::f32::consts::PI;
use StellarMath::algebra::fourier::{fft, ifft};
use StellarMath::structure::ndsignal::{Complex, NdSignal};

fn generate_dummy_series(n: usize) -> Vec<Complex> {
    let mut signal = Vec::with_capacity(n);
    for t in 0..n {
        let measurement = {
            (PI * (t as f32) / 4_f32).cos()
                + (PI * (t as f32) / 5_f32).cos()
                + (PI * (t as f32) / 2_f32).sin()
        };

        signal.push(Complex::new(measurement, 0_f32));
    }
    signal
}

fn pretty_format(data: &[Complex]) -> NdSignal {
    let mut dims = vec![1; 2];
    dims[0] = data.len();
    NdSignal {
        dims,
        data: data.to_vec(),
    }
}

fn main() {
    let k = 16;
    let mut data = generate_dummy_series(k);
    fft(&mut data);
    println!("Development Version {:?}", pretty_format(&data));
    ifft(&mut data);
    println!("Inverse Version {:?}", pretty_format(&data));
}
