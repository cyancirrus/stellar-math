use crate::structure::ndarray::NdArray;

const EPSILON: f32 = 1e-8;

/// LuPivotDecompose
///
/// * n: cardinality
/// * swaps: determines sign of determinant
/// * pivots: how to get back information
/// * matrix: what information is stored

pub struct LuPivotDecompose {
    n: usize,
    swaps: usize,
    pivots: Vec<usize>,
    pub matrix: NdArray,
}
impl LuPivotDecompose {
    pub fn new(mut matrix: NdArray, workspace: &mut [f32]) -> Self {
        // Doolittle
        debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
        let n = matrix.dims[0];
        let m = &mut matrix.data;
        let mut pivots: Vec<usize> = Vec::with_capacity(n);
        let mut swaps = 0;
        let workspace = &mut workspace[..n];
        for k in 0..n {
            let krow = k * n;
            let mut p = k;
            let mut scl = m[krow + k];
            {
                let mut irow = krow;
                for i in k + 1..n {
                    irow += n;
                    let cur_s = m[irow + k];
                    if cur_s.abs() > scl.abs() {
                        p = i;
                        scl = cur_s;
                    }
                }
            }
            pivots.push(p);
            if p != k {
                swaps += 1;
                let prow = p * n;
                workspace.copy_from_slice(&mut m[krow..krow + n]);
                m.copy_within(prow..prow + n, krow);
                m[prow..prow + n].copy_from_slice(&workspace);
            }
            if scl.abs() > EPSILON {
                let mut irow = krow;
                for _ in k + 1..n {
                    irow += n;
                    let v = m[irow + k] / scl;
                    m[irow + k] = v;
                    for j in k + 1..n {
                        m[irow + j] -= v * m[krow + j];
                    }
                }
            }
        }
        Self {
            n,
            swaps,
            pivots,
            matrix,
        }
    }
    pub fn new_dl(mut matrix: NdArray) -> Self {
        // Doolittle
        debug_assert_eq!(matrix.dims[0], matrix.dims[1]);
        let n = matrix.dims[0];
        let mut pivots: Vec<usize> = (0..n).collect();
        let mut swaps = 0;

        let mut val;
        for k in 0..n {
            val = 0.0;
            for i in k..n {
                let mag = matrix.data[i * n + k].abs();
                if mag >= val {
                    pivots[k] = i;
                    val = mag;
                }
            }
            if pivots[k] != k {
                swaps += 1;
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
        Self {
            n,
            swaps,
            pivots,
            matrix,
        }
    }
    pub fn reconstruct(&self) -> NdArray {
        let mut data = vec![0f32; self.n * self.n];
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
    pub fn log_determinant(&self) -> f32 {
        let mut det = 0f32;
        for k in 0..self.n {
            det += self.matrix.data[k * self.n + k].abs().ln();
        }
        if self.swaps & 1 == 1 {
            det = -det
        }
        det
    }
    pub fn condition(&self) -> f32 {
        let (mut max, mut min) = (0f32, f32::MAX);
        for k in 0..self.n {
            let v = self.matrix.data[k * self.n + k].abs();
            max = max.max(v);
            min = min.min(v);
        }
        max / min
    }
    pub fn lower_bound_condition(&self) -> f32 {
        let mut max_col_sum:f32 = 0.0;
        let mut min_uii:f32 = f32::MAX;
        for j in 0..self.n {
            let mut col_sum:f32 = 0.0;
            for i in 0..self.n {
                let v = self.matrix.data[i * self.n + j].abs();
                col_sum += v;
                if i == j {
                    min_uii = min_uii.min(v);
                }
            }
            max_col_sum = max_col_sum.max(col_sum);
        }
        if min_uii == 0.0 {
            return f32::INFINITY
        }
        max_col_sum / min_uii
    }
}
impl LuPivotDecompose {
    pub fn left_apply_l(&self, target: &mut NdArray) {
        // LA = Output
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        debug_assert_eq!(cols, trows);
        let m = &self.matrix.data;
        let mut mrow = rows * cols;
        let mut trow = trows * tcols;
        // lii == 1 => l[0,0] = 1
        for i in (1..rows).rev() {
            trow -= tcols;
            mrow -= cols;
            let m_suffix = &m[mrow..mrow + cols];
            let (tgt_upper, mut tgt_suffix) = target.data.split_at_mut(trow);
            tgt_suffix = &mut tgt_suffix[..tcols];
            {
                let mut krow = 0;
                for k in 0..i {
                    let scalar = m_suffix[k];
                    if scalar.abs() < EPSILON {
                        continue;
                    }
                    let cur_suffix = &tgt_upper[krow..krow + tcols];
                    // cur_suffix = &cur_suffix[..tcols];
                    for j in 0..tcols {
                        tgt_suffix[j] += scalar * cur_suffix[j];
                    }
                    krow += tcols;
                }
            }
        }
    }
    pub fn left_apply_u(&self, target: &mut NdArray) {
        // UA = Output
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        debug_assert_eq!(cols, trows);
        let m = &self.matrix.data;
        let t = &mut target.data;
        let mut moffset = 0;
        let mut toffset = 0;
        for i in 0..rows {
            let m_suffix = &m[moffset..moffset + cols];
            let (tgt_upper, tgt_lower) = t.split_at_mut(toffset + tcols);
            let tgt_suffix = &mut tgt_upper[toffset..toffset + tcols];
            {
                let scalar = m_suffix[i];
                for j in 0..tcols {
                    tgt_suffix[j] *= scalar ;
                }
            }
            {
                let mut koffset = 0;
                for k in i + 1..cols {
                    let t_suffix = &tgt_lower[koffset..koffset + tcols];
                    let scalar = m_suffix[k];
                    for j in 0..tcols {
                        tgt_suffix[j] += scalar * t_suffix[j];
                    }
                    koffset += tcols;
                }
            }
            toffset += tcols;
            moffset += cols;
        }
    }
    // pub fn right_apply_l(&self, target: &mut NdArray, workspace: &mut [f32]) {
    //     // AL = Output
    //     let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
    //     let (trows, tcols) = (target.dims[0], target.dims[1]);
    //     debug_assert_eq!(tcols, rows);
    //     let workspace = &mut workspace[..tcols];
    //     let t = &mut target.data;
    //     let m = &self.matrix.data;
    //     for i in 0..trows {
    //         let outer_suffix = &mut t[i * tcols.. i * tcols + tcols];
    //         for k in 0..rows {
    //             workspace[k] = t[i * tcols + k];
    //             let m_suffix = &m[k * cols.. k * cols + cols];
    //             let scalar = outer_suffix[k];
    //             for j in 0..k {
    //                 workspace[j] += scalar * m_suffix[j];
    //             }
    //         }
    //         outer_suffix.copy_from_slice(workspace);
    //     }
    // }
    pub fn right_apply_l(&self, target: &mut NdArray) {
        // AL = Output
        debug_assert_eq!(target.dims[1], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..trows {
            for j in 0..rows {
                for k in j + 1..rows {
                    target.data[i * tcols + j] +=
                        target.data[i * tcols + k] * self.matrix.data[k * cols + j];
                }
            }
        }
    }
    pub fn right_apply_u(&self, target: &mut NdArray) {
        // AU = Output
        debug_assert_eq!(target.dims[1], self.matrix.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let (trows, tcols) = (target.dims[0], target.dims[1]);
        for i in 0..trows {
            for j in (0..rows).rev() {
                target.data[i * tcols + j] *= self.matrix.data[j * cols + j];
                for k in 0..j {
                    target.data[i * tcols + j] +=
                        target.data[i * tcols + k] * self.matrix.data[k * cols + j];
                }
            }
        }
    }
}

impl LuPivotDecompose {
    pub fn left_apply_l_vec(&self, target: &mut [f32]) {
        // Lx
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for i in (0..rows).rev() {
            for k in 0..i {
                target[i] += self.matrix.data[i * cols + k] * target[k];
            }
        }
    }
    pub fn left_apply_u_vec(&self, target: &mut [f32]) {
        // Ux
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for i in 0..rows {
            target[i] *= self.matrix.data[i * cols + i];
            for k in i + 1..cols {
                target[i] += self.matrix.data[i * cols + k] * target[k];
            }
        }
    }
    pub fn right_apply_l_vec(&self, target: &mut [f32]) {
        //x'L
        debug_assert_eq!(self.matrix.dims[1], target.len());
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        for j in 0..cols {
            for k in j + 1..rows {
                target[j] += target[k] * self.matrix.data[k * cols + j]
            }
        }
    }
    pub fn right_apply_u_vec(&self, target: &mut [f32]) {
        //x'U
        debug_assert_eq!(self.matrix.dims[1], target.len());
        // let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        let cols = self.matrix.dims[1];
        for j in (0..cols).rev() {
            target[j] *= self.matrix.data[j * cols + j];
            for k in 0..j {
                target[j] += target[k] * self.matrix.data[k * cols + j]
            }
        }
    }
}

impl LuPivotDecompose {
    // Ax = b;
    // PA ~ LU;
    // PAx = Pb;
    // LUx = b*;
    // Lz = b*;
    // => z
    // Ux = z;
    // => x
    pub fn solve_inplace(&self, y: &mut NdArray) {
        debug_assert_eq!(self.matrix.dims[1], y.dims[0]);
        self.pivot_inplace(y);
        self.forward_solve_inplace(y);
        self.backward_solve_inplace(y);
    }
    pub fn solve_inplace_vec(&self, y: &mut [f32]) {
        debug_assert_eq!(self.matrix.dims[1], y.len());
        self.pivot_inplace_vec(y);
        self.forward_solve_inplace_vec(y);
        self.backward_solve_inplace_vec(y);
    }
    pub fn pivot_inplace(&self, y: &mut NdArray) {
        let t_cols = y.dims[1];
        for (s, &d) in self.pivots.iter().enumerate() {
            if s != d {
                for k in 0..t_cols {
                    y.data.swap(s * t_cols + k, d * t_cols + k);
                }
            }
        }
    }
    pub fn unpivot_inplace(&self, y: &mut NdArray) {
        let t_cols = y.dims[1];
        for (s, &d) in self.pivots.iter().enumerate().rev() {
            if s != d {
                for k in 0..t_cols {
                    y.data.swap(s * t_cols + k, d * t_cols + k);
                }
            }
        }
    }
    pub fn pivot_inplace_vec(&self, y: &mut [f32]) {
        for (s, &d) in self.pivots.iter().enumerate() {
            if s != d {
                y.swap(d, s);
            }
        }
    }
    pub fn unpivot_inplace_vec(&self, y: &mut [f32]) {
        for (s, &d) in self.pivots.iter().enumerate().rev() {
            if s != d {
                y.swap(d, s);
            }
        }
    }
    // TODO: migrate to ikj, j is inner most loop this iteration doesn't make sense
    pub fn forward_solve_inplace(&self, y: &mut NdArray) {
        // transforms y -> z
        debug_assert_eq!(self.matrix.dims[1], y.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        // let (trows, tcols) = (y.dims[0], y.dims[1]);
        let tcols = y.dims[1];
        for j in 0..tcols {
            for i in 0..rows {
                for k in 0..i {
                    y.data[i * tcols + j] -= self.matrix.data[i * cols + k] * y.data[k * tcols + j];
                }
            }
        }
    }
    pub fn backward_solve_inplace(&self, z: &mut NdArray) {
        // transforms y -> z
        debug_assert_eq!(self.matrix.dims[1], z.dims[0]);
        let (rows, cols) = (self.matrix.dims[0], self.matrix.dims[1]);
        // let (trows, tcols) = (z.dims[0], z.dims[1]);
        let tcols = z.dims[1];
        for i in (0..rows).rev() {
            for k in i + 1..cols {
                for j in 0..tcols {
                    z.data[i * tcols + j] -= self.matrix.data[i * cols + k] * z.data[k * tcols + j];
                }
            }
            for j in 0..tcols {
                z.data[i * tcols + j] /= self.matrix.data[i * cols + i];
            }
        }
    }
    pub fn forward_solve_inplace_vec(&self, y: &mut [f32]) {
        // transforms y -> z
        debug_assert_eq!(self.matrix.dims[1], y.len());
        let cols = self.matrix.dims[1];
        for i in 0..cols {
            for k in 0..i {
                y[i] -= self.matrix.data[i * cols + k] * y[k]
            }
        }
    }
    pub fn backward_solve_inplace_vec(&self, z: &mut [f32]) {
        // transforms z -> x
        debug_assert_eq!(self.matrix.dims[1], z.len());
        let cols = self.matrix.dims[1];
        for i in (0..cols).rev() {
            for k in i + 1..cols {
                z[i] -= self.matrix.data[i * cols + k] * z[k]
            }
            z[i] /= self.matrix.data[i * cols + i]
        }
    }
}
