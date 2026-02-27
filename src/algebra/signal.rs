use crate::structure::ndsignal::Complex;
use std::f32::consts::PI;

fn twiddle(k: f32, n: f32) -> Complex {
    // exp( (-2 * pi * i * k / n)
    // = cos(*) - isin(*)
    // let phase= 2_f32 * PI * k / n;
    let phase = 2_f32 * PI * k / n;
    Complex::new(phase.cos(), -phase.sin())
}

pub fn cooley_tukey(x: &mut [Complex], n: usize, s: usize) {
    let mut p_e: Complex;
    let mut q_e: Complex;
    let mut p_o: Complex;
    let mut q_o: Complex;
    let mut t: Complex;
    println!("HEAD");
    println!("n:{}, s:{}", n, s);
    println!("Static X {:?}", x);
    println!("------------------------------------");
    if n > 1 {
        cooley_tukey(&mut x[..], n / 2, 2 * s);
        for k in 0..n / 2 - 1 {
            // t = twiddle(k as f32 /2_f32, n as f32);
            // t = twiddle((s * 2) as f32 , n as f32);
            t = twiddle(s as f32, n as f32);
            // println!(" k:{}, s:{}, n:{}, t:{}\n", k, s, n, t);
            p_e = x[k];
            q_e = t * x[k + s];
            // println!("inputs! x[{}] and x[{}]", k, k + s);
            // println!("targets! x[{}] and x[{}]", k, k + 4 / 2);
            // println!("X {:?}", x);
            p_o = x[k + n / 2];
            q_o = t * x[k + n / 2 + s];
            // println!("inputs! x[{}] and x[{}]", k + n / 2, n / 2 + s);
            // println!("targets! x[{}] and x[{}]", k + 1, k +  1  + n / 2);
            // println!("targets! x[{}] and x[{}]", k + 1, k + 1 + 4 / 2);
            x[k] = p_e + q_e;
            x[k + 4 / 2] = p_e - q_e;
            x[k + 1] = p_o + q_o;
            x[k + 4 / 2 + 1] = p_o - q_o;
            // println!("p:{}, q:{}", p_e, q_e);
            // println!("p:{}, q:{}", p_o, q_o);
            // println!("X {:?}", x);
            // println!("------------------------------------");
        }
    }
}
