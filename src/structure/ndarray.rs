#![allow(warnings)]
use crate::algebra::math;
use rayon::prelude::*;
use std::fmt;

#[derive(Clone)]
pub struct NdArray {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

impl NdArray {
    pub fn new(dims: Vec<usize>, data: Vec<f32>) -> NdArray {
        NdArray { dims, data }
    }

    pub fn print(&self) {
        let (rows, cols) = (self.dims[0], self.dims[1]);

        // Determine the max width needed for alignment
        let max_width = self
            .data
            .iter()
            .map(|v| format!("{:.3}", v).len())
            .max()
            .unwrap_or(4); // Default width if empty

        let mut output = String::new();
        output.push_str("(\n");
        for i in 0..rows {
            output.push_str("\t(");
            for j in 0..cols {
                let idx = i * cols + j;
                let formatted = format!("{:width$.3}", self.data[idx], width = max_width);
                output.push_str(&formatted);
                if j < cols - 1 {
                    output.push_str(", ");
                }
            }
            output.push_str("),\n");
        }
        output.push(')');

        println!("{}", output);
    }
}

impl fmt::Debug for NdArray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (rows, cols) = (self.dims[0], self.dims[1]);

        // Determine the max width needed for alignment
        let max_width = self
            .data
            .iter()
            .map(|v| format!("{:.3}", v).len())
            .max()
            .unwrap_or(4);

        let mut output = String::new();
        output.push_str("(\n");
        for i in 0..rows {
            output.push_str("\t(");
            for j in 0..cols {
                let idx = i * cols + j;
                let formatted = format!("{:width$.3}", self.data[idx], width = max_width);
                output.push_str(&formatted);
                if j < cols - 1 {
                    output.push_str(", ");
                }
            }
            output.push_str("),\n");
        }
        output.push(')');

        write!(f, "{}", output)
    }
}
