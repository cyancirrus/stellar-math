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
    pub fn dims(&self) -> &Vec<usize> {
        &self.dims
    }
    pub fn card(&self) -> usize {
        self.data.len()
    }
    pub fn clear(&mut self) {
        self.data = vec![0_f32; self.data.len()];
    }
    pub fn diff(&mut self, other: Self) {
        debug_assert!(other.dims() == self.dims());
        for idx in 0..self.card() {
            self.data[idx] -= other.data[idx];
        }
    }
    pub fn sum(&mut self, other: Self) {
        debug_assert!(other.dims() == self.dims());
        for idx in 0..self.dims.len() {
            self.data[idx] += other.data[idx];
        }
    }
}

// const EPSILON:f32 = 1e-6;
// impl PartialEq for NdArray {
//     fn eq(&self, other:&Self) -> bool {
//         let c = self.dims.len();
//         if self.dims != other.dims {
//             return false;
//         }
//         if self.data.len() != other.data.len() {
//             return false;
//         }
//         for i in 0.. self.data.len() {
//             if (self.data[i] - other.data[i]).abs() > EPSILON {
//                 return false;
//             }
//         }
//         true
//     }
// }

struct NdIterator<'a> {
    drow: usize,
    dcol: usize,
    row: usize,
    col: usize,
    ndarray: &'a NdArray,
}

impl<'a> Iterator for NdIterator<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let (rows, cols) = (self.ndarray.dims[0], self.ndarray.dims[1]);
        (self.row, self.col) = (self.row + self.drow, self.col + self.dcol);

        if self.row < rows && self.col < cols {
            Some(self.ndarray.data[self.row * cols + self.col])
        } else {
            None
        }
    }
}

impl NdArray {
    fn iterate(&self, drow: usize, dcol: usize, row: usize, col: usize) -> NdIterator {
        NdIterator {
            drow,
            dcol,
            row,
            col,
            ndarray: self,
        }
    }
    pub fn transpose_square(&mut self) {
        debug_assert_eq!(self.dims[0], self.dims[1]);
        let n = self.dims[0];
        for i in 0..n {
            for j in i + 1..n {
                self.data.swap(i * n + j, j * n + i);
            }
        }
    }
    pub fn transpose(&self) -> NdArray {
        let mut dims = self.dims.clone();
        let mut data = vec![0_f32; self.card()];
        dims.swap(0, 1);
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                data[i * dims[1] + j] = self.data[j * dims[0] + i]
            }
        }
        NdArray { dims, data }
    }
}

impl NdArray {
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
