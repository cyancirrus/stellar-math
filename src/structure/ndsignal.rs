use rayon::prelude::ParallelIterator;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::f32::consts::PI;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Debug, Clone, Copy)]
pub struct Complex {
    r: f32,
    i: f32,
}

#[derive(Clone)]
pub struct NdSignal {
    pub dims: Vec<usize>,
    pub data: Vec<Complex>,
}

impl NdSignal {
    pub fn new(dims: Vec<usize>, data: Vec<Complex>) -> Self {
        Self { dims, data }
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, other: Complex) {
        self.r += other.r;
        self.i += other.i
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.r + other.r, self.i + other.i)
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, other: Complex) {
        let r = self.r * other.r - self.i * other.i;
        let i = self.r * other.i + other.r * self.i;
        self.r = r;
        self.i = i;
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Complex) -> Complex {
        let r = self.r * other.r - self.i * other.i;
        let i = self.r * other.i + other.r * self.i;
        Complex::new(r, i)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.r - other.r, self.i - other.i)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, other: Complex) {
        self.r = self.r - other.r;
        self.i = self.i - other.i;
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the real and imaginary parts with a fixed width for alignment
        write!(f, "( r:{:<6.3}, i:{:<6.3})", self.r, self.i)
    }
}

impl Complex {
    pub fn new(r: f32, i: f32) -> Self {
        Complex { r, i }
    }
    pub fn add(x: Complex, y: Complex) -> Complex {
        Complex::new(x.r + y.r, x.i + y.i)
    }
    pub fn product(x: Complex, y: Complex) -> Complex {
        Complex::new(x.r * y.r, -x.i * y.i)
    }
    pub fn multiply(x: Complex, y: Complex) -> Complex {
        Complex::new(x.r * y.r - x.i * y.i, x.r * y.i + x.i * y.r)
    }
    pub fn exponent(x: Complex) -> f32 {
        //e(-ix) = cos(x) - isin(x)
        //e((a + bi)) = exp(a)(cos(x) - i sin(x))
        //isin(x) = (exp(x) - exp(-x)) / 2
        x.r.exp() * (x.i.exp() - (-x.i).exp()) / 2_f32
    }
    pub fn zero() -> Complex {
        Complex::new(0_f32, 0_f32)
    }
}

impl Complex {
    pub fn scale(&mut self, alpha: f32) {
        self.r *= alpha;
        self.i *= alpha;
    }
}

impl NdSignal {
    pub fn print(&self) {
        let (rows, cols) = (self.dims[0], self.dims[1]);

        // Determine the max width needed for alignment
        let max_width = self
            .data
            .iter()
            .map(|v| format!("{}", v).len()) // Use the Display trait
            .max()
            .unwrap_or(8); // Default width if empty

        let mut output = String::new();
        output.push_str("(\n");
        for i in 0..rows {
            output.push_str("\t");
            for j in 0..cols {
                let idx = i * cols + j;
                let formatted = format!("{:width$}", self.data[idx], width = max_width); // Format with the determined width
                output.push_str(&formatted);
                if j < cols - 1 {
                    output.push_str(", \t");
                }
            }
            output.push_str(",\n");
        }
        output.push(')');

        println!("{}", output);
    }
}

impl fmt::Debug for NdSignal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (rows, cols) = (self.dims[0], self.dims[1]);

        // Determine the max width needed for alignment
        let max_width = self
            .data
            .iter()
            .map(|v| format!("{}", v).len()) // Use the Display trait here
            .max()
            .unwrap_or(8);

        let mut output = String::new();
        output.push_str("(\n");
        for i in 0..rows {
            output.push_str("\t");
            for j in 0..cols {
                let idx = i * cols + j;
                let formatted = format!("{:width$}", self.data[idx], width = max_width); // Format with the determined width
                output.push_str(&formatted);
                if j < cols - 1 {
                    output.push_str(", \t");
                }
            }
            output.push_str(",\n");
        }
        output.push(')');

        write!(f, "{}", output)
    }
}
