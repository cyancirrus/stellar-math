use crate::algebra::ndmethods::tensor_mult;
use crate::decomposition::lu::LuDecomposition;
use crate::random::generation::generate_random_matrix;
use crate::structure::ndarray::NdArray;

pub struct LuPivotDecomp {
    n: usize,
    pivots: Vec<usize>,
    matrix: NdArray,
}
impl LuPivotDecomp {
    pub fn new(mut matrix: NdArray) -> Self {
        // Croute
        // A[j, *] = c *A[i, *]
        // => c = A[i,j] / A[j,j]
        debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
        let n = matrix.dims[0];
        let mut pivots = vec![usize::MAX; n];

        // for lij, we need knowledge of ujj due to formula
        // this means that we need u[0..i.min(j)] upper triangular calculated
        for i in 0..n {
            let mut val = 0.0;
            for k in i..n {
                let mag = matrix.data[k * n + i].abs();
                if mag >= val {
                    pivots[i] = k;
                    val = mag;
                }
            }
            for k in 0..n {
                matrix.data.swap(i * n + k, pivots[i] * n + k);
            }
            for j in 0..n {
                for k in 0..i.min(j) {
                    matrix.data[i * n + j] -= matrix.data[i * n + k] * matrix.data[k * n + j]
                }
                if i > j {
                    matrix.data[i * n + j] /= matrix.data[j * n + j];
                }
            }
        }
        Self { n, pivots, matrix }
    }
    pub fn new_row(mut matrix: NdArray) -> Self {
        // Doolittle
        debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
        let n = matrix.dims[0];
        let mut pivots: Vec<usize> = (0..n).collect();

        let mut val;
        for k in 0..n {
            val = 0.0;
            for i in k + 1..n {
                let mag = matrix.data[i * n + k].abs();
                if mag >= val {
                    pivots[k] = i;
                    val = mag;
                }
            }
            if pivots[k] != k {
                for j in 0..n {
                    matrix.data.swap(k * n + j, pivots[k] * n + j);
                }
            }
            for i in k + 1..n {
                matrix.data[i * n + k] /= matrix.data[k * n + k];
            }
            for i in k + 1..n {
                for j in k + 1..n {
                    matrix.data[i * n + j] -= matrix.data[i * n + k] * matrix.data[k * n + j]
                }
            }
        }
        Self { n, pivots, matrix }
    }
    pub fn left_apply_u(&self, mut x: NdArray) -> NdArray {
        // UA => Output
        for i in 0..self.n {
            for j in 0..self.n {
                x.data[i * self.n + j] *= self.matrix.data[i * self.n + i];
                for k in i + 1..self.n {
                    x.data[i * self.n + j] +=
                        self.matrix.data[i * self.n + k] * x.data[k * self.n + j]
                }
            }
        }
        x
    }
    pub fn reconstruct(&self) -> NdArray {
        let mut data = vec![0_f32; self.n * self.n];
        let dims = vec![self.n; 2];
        for i in 0..self.n {
            for j in 0..self.n {
                // if i > j then we hit an add at a[i][k] b/c diagonal is identity;
                for k in 0..=i.min(j) {
                    if k == i {
                        data[i * self.n + j] += self.matrix.data[i * self.n + j];
                    } else {
                        data[i * self.n + j] +=
                            self.matrix.data[i * self.n + k] * self.matrix.data[k * self.n + j];
                    }
                }
            }
        }
        for (n, &k) in self.pivots.iter().enumerate().rev() {
            if n == k {
                continue;
            }
            for j in 0..self.n {
                data.swap(n * self.n + j, k * self.n + j);
            }
        }
        NdArray { dims, data }
    }
}

// a [4] [ 2]
// 1 -> 2, 2 -> 3, 3 -> 1

// fn test_reconstruct() {
//     let n = 4;
//     let x = generate_random_matrix(n, n);
//     println!("x {x:?}");
//     let lu = LuPivotDecomp::new(x.clone());
//     // let lu = LuDecomposition::new(x.clone());
//     let out = lu.reconstruct();
//     assert_eq!(x, out);
// }

// fn main() {
//     test_reconstruct();
// }
