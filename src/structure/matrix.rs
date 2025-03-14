#![allow(warnings)]
use crate::algebra::math;
use rayon::prelude::*;
use std::fmt;

pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        assert!(rows > 0, "rows is not greater than or equal to 0");
        assert!(cols > 0, "cols is not greater than or equal to 0");
        assert_eq!(data.len(), rows * cols, "dimension mismatch in matrix");
        Matrix { rows, cols, data }
    }
}
