use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;

pub struct QrDecomposition {
    pub projections: Vec<HouseholderReflection>,
    pub triangle: NdArray,
}

pub fn qr_decompose(mut x: NdArray) -> QrDecomposition {
    let rows = x.dims[0];
    let cols = x.dims[1];
    let mut projections = Vec::with_capacity(cols.min(rows));

    for o in 0..cols.min(rows) {
        let column_vector = (o..rows)
            .into_par_iter()
            .map(|r| x.data[r * cols + o])
            .collect::<Vec<f32>>();
        let householder = householder_params(&column_vector);
        projections.push(householder);
        let mut queue: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o) * (rows - o)];
        for i in 0..(rows - o).min(cols - o) {
            for j in 0..cols - o {
                // Need to compute the change for everything to the right of the initial vector
                if i <= j || j > o {
                    let sum = (0..rows - o)
                        .into_par_iter()
                        .map(|k| {
                            x.data[(k + o) * cols + (j + o)]
                                * projections[o].beta
                                * projections[o].vector[i]
                                * projections[o].vector[k]
                        })
                        .sum();
                    queue[i * (cols - o) + j].0 = (i + o) * cols + (j + o);
                    queue[i * (cols - o) + j].1 = sum;
                }
            }
        }
        queue.iter().for_each(|q| x.data[q.0] -= q.1);
        (o + 1..rows).for_each(|i| x.data[i * cols + o] = 0_f32);
    }
    QrDecomposition::new(projections, x)
}

impl QrDecomposition {
    pub fn new(projections: Vec<HouseholderReflection>, triangle: NdArray) -> Self {
        Self {
            projections,
            triangle,
        }
    }
    pub fn left_multiply(&self, other: NdArray) -> NdArray {
        let dims = self.triangle.dims.clone();
        let mut data = other.data.clone();
        let rows = other.dims[0];

        (0..other.dims[0]).for_each(|i| {
            let start = i * rows;
            let end = (i + 1) * rows;
            let row = &data[start..end];
            let cordinate = self.determine_basis(row.to_vec());
            for k in 0..cordinate.len() {
                // If you want to generate columns
                // data[k * dims[0] + i] = cordinate[k];
                data[i * dims[0] + k] = cordinate[k];
            }
        });
        NdArray::new(dims, data)
    }
    pub fn triangle_rotation(&self) -> NdArray {
        let dims = self.triangle.dims.clone();
        let mut data = self.triangle.data.clone();
        let rows = self.triangle.dims[0];

        (0..self.triangle.dims[0]).rev().for_each(|i| {
            let start = i * rows;
            let end = (i + 1) * rows;
            let row = &data[start..end];
            let cordinate = self.determine_basis(row.to_vec());
            for k in 0..cordinate.len() {
                data[i * dims[0] + k] = cordinate[k];
            }
        });
        NdArray::new(dims, data)
    }
    fn determine_basis(&self, mut data: Vec<f32>) -> Vec<f32> {
        for i in 0..self.projections.len() {
            let mut delta = vec![0_f32; self.triangle.dims[0]];
            let projection = &self.projections[i];
            for j in 0..projection.vector.len() {
                for k in 0..projection.vector.len() {
                    delta[i + j] -=
                        projection.beta * projection.vector[k] * projection.vector[j] * data[i + k];
                }
            }
            for j in 0..delta.len() {
                data[j] += delta[j];
            }
        }
        data
    }
}
