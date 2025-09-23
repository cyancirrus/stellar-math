use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use crate::algebra::vector::dot_product;
use crate::algebra::ndmethods::create_identity_matrix;
use rayon::prelude::*;

#[derive(Debug)]
pub struct QrDecomposition {
    pub projections: Vec<HouseholderReflection>,
    pub triangle: NdArray,
}

pub fn qr_decompose(mut x: NdArray) -> QrDecomposition {
    // TODO : currently this runs for 0..cols.min(rows) should run to 0..cols.min(rows) - 1
    // direct depends: left_multiply, projection_matrix, schur
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
        let mut update: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o) * (rows - o)];
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
                    update[i * (cols - o) + j].0 = (i + o) * cols + (j + o);
                    update[i * (cols - o) + j].1 = sum;
                }
            }
        }
        update.iter().for_each(|q| x.data[q.0] -= q.1);
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
    pub fn size(&self) -> usize {
        self.projections.len()
    }
    pub fn projection_matrix(&self) -> NdArray {
        let size = self.size();
        let mut matrix = create_identity_matrix(size);
        let mut w: Vec<f32> = vec![0_f32; size ];
        // I - Buu'
        // H[i+1] * H[i] = H[i+1] - B[i](H[i+1]u[i])u'[i]
        // Hu := w
        // H[i+1] -= B[i] *w[i+1]u'[i]
        // TODO: This should coincide with the change in the for 0..cols.min(rows)-1 change
        for p in 0..size - 1 {
            let proj = &self.projections[p];
            println!("w[i] {:?}", w);
            for i in p..size {
                for j in p..size {
                    w[i] += matrix.data[i * size + j] * proj.vector[j - p];
                }
                println!("beta {:?}", proj.beta);
                w[i] *= proj.beta;
            }
            println!("w[i] {:?}", w);
            for i in p..size {
                for j in p..size {
                    matrix.data[i * size + j] -= w[i] * proj.vector[j - p];
                }
                w[i] = 0_f32;
            }
            println!("projection {matrix:?}");
        }
        matrix
    }
    // pub fn deothrogonalize(&self, v: &mut Vec<f32>) {
    //     let vlen = v.len();
    //     for h in self.projections.iter().rev() {
    //         if h.beta == 0_f32 {continue;}
    //         let dot = dot_product(&h.vector, &v);
    //         for i in 0..h.vector.len() {
    //             v[vlen -1 - i] -= h.beta * h.vector[i] * dot;
    //         }
    //     }
    // }
    pub fn left_multiply(&self, target: &mut NdArray) {
        // H[i]*X = X - Buu'X
        // w = u'X
        debug_assert!(target.dims[0] == target.dims[1]);
        debug_assert!(target.dims[0] == self.size());
        let (rows, cols) = (target.dims[0], target.dims[1]);
        let mut w = vec![0_f32; rows];
        // TODO: Only iterate up to that version
        for p in 0..self.size() - 1 {
            let proj = &self.projections[p];
            for j in 0..cols {
                for i in p..rows {
                    w[j] += proj.vector[i - p] * target.data[ i * cols + j ];
                }
            }
            for j in 0..cols {
                for i in p..rows {
                    target.data[ i * cols + j ] -= proj.beta * w[ j ] * proj.vector[ i - p ];
                }
                w[j] = 0_f32;
            }
        }
    }
    // pub fn left_multiply(&self, other: NdArray) -> NdArray {
    //     let dims = self.triangle.dims.clone();
    //     let mut data = other.data.clone();
    //     let rows = other.dims[0];

    //     (0..other.dims[0]).for_each(|i| {
    //         let start = i * rows;
    //         let end = (i + 1) * rows;
    //         let row = &data[start..end];
    //         let cordinate = self.multiply_vector(row.to_vec());
    //         for k in 0..cordinate.len() {
    //             // If you want to generate columns
    //             // data[k * dims[0] + i] = cordinate[k];
    //             data[i * dims[0] + k] = cordinate[k];
    //         }
    //     });
    //     NdArray::new(dims, data)
    // }
    // pub fn triangle_rotation(&self) -> NdArray {
    //     self.triangle.clone()
    //     // // skeptical
    //     // let dims = self.triangle.dims.clone();
    //     // let mut data = self.triangle.data.clone();
    //     // let rows = self.triangle.dims[0];

    //     // (0..self.triangle.dims[0]).rev().for_each(|i| {
    //     //     let start = i * rows;
    //     //     let end = (i + 1) * rows;
    //     //     let row = &data[start..end];
    //     //     let cordinate = self.multiply_vector(row.to_vec());
    //     //     for k in 0..cordinate.len() {
    //     //         data[i * dims[0] + k] = cordinate[k];
    //     //     }
    //     // });
    //     // NdArray::new(dims, data)
    // }
    fn multiply_vector(&self, mut data: Vec<f32>) -> Vec<f32> {
        let size = self.size(); 
        debug_assert!(data.len() == size);
        // H[i+1]x = (I - buu')x  = x - b*u*(u'x)
        for p in 0..size {
            let mut scalar = 0_f32;
            let proj = &self.projections[p];
            debug_assert!(size == proj.vector.len() + p);
            for i in 0..size-p {
                scalar += data[ i + p ] * proj.vector[i];
            }
            for i in 0..size-p {
                data[ i + p ] -= scalar * proj.beta * proj.vector[i];
            }
        }
        data
    }
}
