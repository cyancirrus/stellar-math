use crate::decomposition::householder::{householder_params, HouseholderReflection};
use crate::structure::ndarray::NdArray;
use crate::algebra::vector::dot_product;
use crate::algebra::ndmethods::create_identity_matrix;
use rayon::prelude::*;

#[derive(Debug)]
pub struct QrDecomposition {
    pub cardinality: usize,
    pub projections: Vec<HouseholderReflection>,
    pub triangle: NdArray,
}

pub fn qr_decompose(mut x: NdArray) -> QrDecomposition {
    // direct depends: left_multiply, projection_matrix, schur
    let (rows, cols)  = (x.dims[0], x.dims[1]);
    let cardinality = rows.min(cols);
    let mut projections = Vec::with_capacity(cols.min(rows));
    let mut update: Vec<f32> = vec![0_f32; rows * cols];

    for o in 0..cardinality.saturating_sub(1) {
        let column_vector = (o..rows)
            .into_par_iter()
            .map(|r| x.data[r * cols + o])
            .collect::<Vec<f32>>();
        let proj = householder_params(&column_vector);
        println!("projection {proj:?}");
        projections.push(proj);
        for i in o..rows {
            for j in o..cols {
                // Need to compute the change for everything to the right of the initial vector
                for k in o..cols {
                    update[ i * cols + j ] += {
                        x.data[ i * cols + k ] *
                        projections[o].vector[k - o] * 
                        projections[o].vector[j - o] *
                        projections[o].beta
                    }
                }
            }
        }
        for i in o..rows {
            for j in o..cols {
                x.data[ i * cols + j ] -= update[ i * cols + j ];
                update[ i * cols + j ] = 0_f32;
            }
        }
    }
    for i in 1..rows {
        for j in 0..i {
            x.data[ i * cols + j ] = 0_f32
        }
    }
    QrDecomposition::new(cardinality, projections, x)
}

// pub fn qr_decompose(mut x: NdArray) -> QrDecomposition {
//     // TODO : currently this runs for 0..cols.min(rows) should run to 0..cols.min(rows) - 1
//     // direct depends: left_multiply, projection_matrix, schur
//     let rows = x.dims[0];
//     let cols = x.dims[1];
//     let mut projections = Vec::with_capacity(cols.min(rows));

//     for o in 0..cols.min(rows) {
//         let column_vector = (o..rows)
//             .into_par_iter()
//             .map(|r| x.data[r * cols + o])
//             .collect::<Vec<f32>>();
//         let householder = householder_params(&column_vector);
//         projections.push(householder);
//         let mut update: Vec<(usize, f32)> = vec![(0, 0_f32); (cols - o) * (rows - o)];
//         for i in 0..(rows - o).min(cols - o) {
//             for j in 0..cols - o {
//                 // Need to compute the change for everything to the right of the initial vector
//                 if i <= j || j > o {
//                     let sum = (0..rows - o)
//                         .into_par_iter()
//                         .map(|k| {
//                             x.data[(k + o) * cols + (j + o)]
//                                 * projections[o].beta
//                                 * projections[o].vector[i]
//                                 * projections[o].vector[k]
//                         })
//                         .sum();
//                     update[i * (cols - o) + j].0 = (i + o) * cols + (j + o);
//                     update[i * (cols - o) + j].1 = sum;
//                 }
//             }
//         }
//         update.iter().for_each(|q| x.data[q.0] -= q.1);
//         (o + 1..rows).for_each(|i| x.data[i * cols + o] = 0_f32);
//     }
//     QrDecomposition::new(projections, x)
// }

impl QrDecomposition {
    pub fn new(cardinality:usize, projections: Vec<HouseholderReflection>, triangle: NdArray) -> Self {
        Self {
            cardinality,
            projections,
            triangle,
        }
    }
    pub fn size(&self) -> usize {
        self.cardinality - 1
    }
    pub fn projection_matrix(&self) -> NdArray {
        let card = self.cardinality;
        let mut matrix = create_identity_matrix(card);
        let mut w: Vec<f32> = vec![0_f32; card ];
        // I - Buu'
        // H[i+1] * H[i] = H[i+1] - B[i](H[i+1]u[i])u'[i]
        // Hu := w
        // H[i+1] -= B[i] *w[i+1]u'[i]
        // TODO: This should coincide with the change in the for 0..cols.min(rows)-1 change
        for p in 0..card.saturating_sub(1) {
            let proj = &self.projections[p];
            for i in p..card {
                for j in p..card {
                    w[ i ] += matrix.data[ i * card + j ] * proj.vector[ j - p ];
                }
                w[ i ] *= proj.beta;
            }
            for i in p..card {
                for j in p..card {
                    matrix.data[ i * card + j ] -= w[ i ] * proj.vector[ j - p ];
                }
                w[ i ] = 0_f32;
            }
        }
        matrix
    }
    pub fn triangle_rotation(&mut self) {
        // Specifically for the Schur algorithm
        // A' = Q'AQ = Q'(QR)Q = RQ
        let card = self.cardinality;
        let mut w: Vec<f32> = vec![0_f32; card]; 
        for p in 0..self.cardinality.saturating_sub(1) {
            let proj = &self.projections[p];
            for i in p..card {
                for j in p..card {
                    w[ i ] += self.triangle.data[ i * card + j ] * proj.vector[ j - p ];
                }
                w[ i ] *= proj.beta;
            }
            for i in p..card {
                for j in p..card {
                    self.triangle.data[ i * card + j ] -= w[ i ] * proj.vector[ j - p ]; 
                }
                w[ i ] = 0_f32;
            }
        }
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
        // AX -> QX
        // H[i]*X = X - Buu'X
        // w = u'X
        debug_assert!(target.dims[0] == target.dims[1]);
        debug_assert!(target.dims[0] == self.cardinality);
        let (rows, cols, card) = (target.dims[0], target.dims[1], self.cardinality);
        let mut w = vec![0_f32; rows];
        // TODO: Only iterate up to that version
        for p in 0..card.saturating_sub(1) {
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
    // pub fn triangle_rotation(&mut self)  {
    //     // // skeptical
    //     let dims = self.triangle.dims.clone();
    //     let mut data = self.triangle.data.clone();
    //     let rows = self.triangle.dims[0];

    //     (0..self.triangle.dims[0]).rev().for_each(|i| {
    //         let start = i * rows;
    //         let end = (i + 1) * rows;
    //         let row = &data[start..end];
    //         let cordinate = self.multiply_vector(row.to_vec());
    //         for k in 0..cordinate.len() {
    //             data[i * dims[0] + k] = cordinate[k];
    //         }
    //     });
    //     self.triangle = NdArray::new(dims, data);
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
