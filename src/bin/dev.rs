#![allow(warnings)]
#[cfg(target_arch = "x86_64")]
use std::f32::consts::PI;
use StellarMath::algebra::fourier::{fft, ifft};
use StellarMath::structure::ndsignal::{Complex, NdSignal};

// fn generate_dummy_series(n: usize) -> Vec<Complex> {
//     let mut signal = Vec::with_capacity(n);
//     for t in 0..n {
//         let measurement = {
//             (PI * (t as f32) / 4_f32).cos()
//                 + (PI * (t as f32) / 5_f32).cos()
//                 + (PI * (t as f32) / 2_f32).sin()
//         };

//         signal.push(Complex::new(measurement, 0_f32));
//     }
//     signal
// }

// fn pretty_format(data: &[Complex]) -> NdSignal {
//     let mut dims = vec![1; 2];
//     dims[0] = data.len();
//     NdSignal {
//         dims,
//         data: data.to_vec(),
//     }
// }

fn debug_data() -> Vec<usize> {
    // dimensions indexed by (x,y,z)
    vec![
        0, 2,
        4, 6,
        //---
        1, 3,
        5, 7 
    ]
}

fn transpose(data:&mut [usize]) {
    let mut dims = vec![2,2,2];
    // static for prototyping
    let (x,y,z) = (2,2,2);
    let mut t = 0;
    for i in 0..x {
        for j in i..y {
            for k in j..z {
                let a = i*(x*y) + j*y + k;
                let b = k*(x*y) + i*y + j;
                let c = j*(x*y) + k*y + i;
                // a, b, c :: 
                // c, b, a :: swap(a, c)
                // b, c, a :: swap(b, c)
                println!(" a: {a:03b}, b: {b:03b}, c: {c:03b}");
                data.swap(
                    a,
                    c,
                );
                data.swap(
                    b,
                    c,
                );
            }
        }
    }
}


fn pretty_print(data:&[usize]) {
    println!("----------");
    for i in (0..8).step_by(2) {
        if i == 4 {
            println!(" - - - - - -");
        }
        println!("[ {:03b}, {:03b} ]", data[i], data[i+1]);
    }
    println!("----------");
}

fn debug_transpose(data:&mut [usize]) {
    transpose(data);
    pretty_print(data);
}

fn main() {
    let mut data = debug_data();
    pretty_print(&data);
    println!("");
    let mut test = data.clone();
    debug_transpose(&mut test);
    debug_transpose(&mut test);
    debug_transpose(&mut test);
    assert_eq!(test, data);
}


// fn main() {
//     // let k = 16;
//     // let mut data = generate_dummy_series(k);
//     // fft(&mut data);
//     // println!("Development Version {:?}", pretty_format(&data));
//     // ifft(&mut data);
//     // println!("Inverse Version {:?}", pretty_format(&data));
// }



// x,y,z -> y,z,x -> z,x,y
//
//
// y11 y12, ...
// a[y * (rows  * cols) + z * cols + x] = a[x * (rows * cols) y * cols + z]
