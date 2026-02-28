use crate::structure::ndarray::NdArray;

pub struct Cholesky {
    pub l: NdArray,
}

const EPSILON:f32 = 1e-6;

impl Cholesky {
    pub fn new(a: NdArray) -> Self {
        // update lij
        // aii = lik * lik + lii^2
        // => lii = (aii - sum 0..i lik)^0.5
        // aij = Sum 0..j lik * ljk + lij * ljj
        // => ljj = 1/ ljj * (aij - Sum 0..j lik * ljk )
        // a32 = l31 * l21 + l32 * l22
        let (rows, cols) = (a.dims[0], a.dims[1]);
        debug_assert_eq!(rows, cols);
        let mut l = vec![0_f32; rows * cols];
        let mut sum ;
        for i in 0..rows {
            let i_row = i * cols;
            let a_row = &a.data[i_row..=i_row + i];
            for j in 0..=i {
                sum = 0_f32;
                let j_row = j * cols;
                for k in 0..j {
                    sum += l[i_row + k] * l[j_row + k];
                }
                let diff = a_row[j] - sum;
                if i == j {
                    l[i_row + j] = diff.max(EPSILON).sqrt()
                } else {
                    let v = diff / l[j_row + j];
                    l[i_row + j] = v;
                    l[j_row + i] = v;
                };
            }
        }
        Self {
            l: NdArray {
                dims: vec![rows, cols],
                data: l,
            },
        }
    }
}
// impl Cholesky {
//     pub fn new(a: NdArray) -> Self {
//         // update lij
//         // aii = lik * lik + lii^2
//         // => lii = (aii - sum 0..i lik)^0.5
//         // aij = Sum 0..j lik * ljk + lij * ljj
//         // => ljj = 1/ ljj * (aij - Sum 0..j lik * ljk )
//         // a32 = l31 * l21 + l32 * l22
//         let (rows, cols) = (a.dims[0], a.dims[1]);
//         debug_assert_eq!(rows, cols);
//         let mut l = vec![0_f32; rows * cols];
//         let mut sum ;
//         for i in 0..rows {
//             let i_row = i * cols;
//             let a_row = &a.data[i_row..=i_row + i];
//             for j in 0..=i {
//                 sum = 0_f32;
//                 let j_row = j * cols;
//                 for k in 0..j {
//                     sum += l[i_row + k] * l[j_row + k];
//                 }
//                 let diff = a_row[j] - sum;
//                 l[i_row + j] = if i == j {
//                     diff.max(EPSILON).sqrt()
//                 } else {
//                     diff / l[j_row + j]
//                 };
//             }
//         }
//         Self {
//             l: NdArray {
//                 dims: vec![rows, cols],
//                 data: l,
//             },
//         }
//     }
// }
