#![allow(dead_code)]
use rand::Rng;
use rand::rngs::ThreadRng;
use rand::distr::StandardUniform;
use crate::decomposition::lu::{lu_decompose, LuDecomposition};
use crate::random::generation::{generate_random_matrix, generate_random_vector, generate_zero_matrix};
use crate::algebra::vector::dot_product;
use crate::structure::ndarray::NdArray;



const CONVERGENCE_CONDITION: f32 = 1e-6;
const EPSILON: f32 = 1e-3;

// NOTE: NOT PRODUCTION READY OR TESTED

pub struct GaussianMixtureModel {
    centroids:usize,
    cardinality:usize,
    pub mixtures:Vec<f32>,
    pub means:Vec<Vec<f32>>,
    pub variance: Vec<NdArray>,
}

fn gaussian(x_bar:&mut Vec<f32>, z_buf:&mut Vec<f32>, det:f32, lu:&LuDecomposition) -> f32 {
    // xbar := x - mean;
    // we have x'Vx, where V := 1/ self.variance
    // solve sub problem LUx = z*; for z* and then <x, z*>
    debug_assert_eq!(x_bar.to_vec(), z_buf.to_vec());
    let card = x_bar.len();
    lu.solve_inplace_vec(z_buf);
    let scaling = -dot_product(&z_buf, &x_bar) / 2_f32;
    scaling.exp() / (2_f32 * std::f32::consts::PI).powf(card as f32 / 2f32) * det.sqrt()
}

fn ln_gaussian(x_bar:&mut Vec<f32>, z_buf:&mut Vec<f32>, det:f32, lu:&LuDecomposition) -> f32 {
    // xbar := x - mean;
    // we have x'Vx, where V := 1/ self.variance
    // solve sub problem LUx = z*; for z* and then <x, z*>
    debug_assert_eq!(x_bar.to_vec(), z_buf.to_vec());
    let card = x_bar.len();
    lu.solve_inplace_vec(z_buf);
    let scaling = dot_product(&z_buf, &x_bar) / 2_f32;
    {
        -(card as f32 / 2f32) * (2f32 * std::f32::consts::PI).ln()
        - 0.5f32 * det.ln()
        - scaling
    }
}
fn initialize_distribution(n:usize, rng:&mut ThreadRng) -> Vec<f32> {
    (0..n).map(|_| rng.sample(StandardUniform)).collect()
}

impl GaussianMixtureModel {
    pub fn new(centroids:usize, cardinality:usize) -> Self {
        Self {
            centroids,
            cardinality:cardinality,
            mixtures:vec![1_f32 / centroids as f32; centroids],
            means: (0..centroids).map(|_| generate_random_vector(cardinality)).collect(),
            variance: (0..centroids).map(|_| generate_random_matrix(cardinality, cardinality)).collect(),
        }
    }
    pub fn expectation_maximization(&mut self, data:&[Vec<f32>]) {
        let mut sum_linear = vec![vec![0_f32; self.cardinality]; self.centroids];
        let mut sum_squares = vec![generate_zero_matrix(self.cardinality, self.cardinality); self.centroids];
        
        let n = data.len();
        let mut x_bar = vec![0_f32; self.cardinality];
        let mut nweighted = vec![0_f32; self.centroids];
        let mut probs = vec![0_f32; self.centroids];
        let mut lus = Vec::with_capacity(self.centroids);
        let mut dets = Vec::with_capacity(self.centroids);
        let mut z_buf= vec![0_f32; self.cardinality];
        for k in 0..self.centroids {
            let lu = lu_decompose(self.variance[k].clone());
            dets.push(lu.find_determinant());
            lus.push(lu);
        }
        for x_i in data {
            let mut max_ln_prob = f32::MIN;
            for k in 0..self.centroids {
                for c in 0..self.cardinality {
                    let val = x_i[c] - self.means[k][c];
                    x_bar[c] = val;
                    z_buf[c] = val;
                }
                probs[k] = self.mixtures[k].ln() + ln_gaussian(&mut x_bar, &mut z_buf, dets[k], &lus[k]);
                max_ln_prob  = max_ln_prob.max(probs[k]);
            }
            let mut scaler = EPSILON;
            for k in 0..self.centroids {
                probs[k] = (probs[k] - max_ln_prob).exp().max(EPSILON);
                scaler += probs[k];
            }
            for k in 0..self.centroids {
                let pr = probs[k] / scaler;
                nweighted[k] += pr;
                for c in 0..self.cardinality {
                    sum_linear[k][c] += pr * x_i[c];
                }
                for i in 0..self.cardinality {
                    for j in 0..=i {
                        sum_squares[k].data[i * self.cardinality + j] += pr * x_i[i] * x_i[j];
                    }
                }
            }
        }
        for k in 0..self.centroids {
            self.mixtures[k] = nweighted[k] / n as f32;
            for c in 0..self.cardinality {
                sum_linear[k][c] /= nweighted[k];
            }
            for i in 0..self.cardinality {
                for j in 0..=i {
                    sum_squares[k].data[i * self.cardinality + j] /= nweighted[k] + EPSILON;
                    sum_squares[k].data[i * self.cardinality + j] -= sum_linear[k][i] * sum_linear[k][j];
                    sum_squares[k].data[j * self.cardinality + i] = sum_squares[k].data[i * self.cardinality + j];
                }
            }
        }
        self.means = sum_linear;
        self.variance = sum_squares;
    }
    fn delta(&self, prev:&[Vec<f32>], curr:&[Vec<f32>]) -> f32 {
        let mut delta = 0_f32;
        for cidx in 0..self.centroids {
            for didx in 0..self.cardinality {
                delta += (prev[cidx][didx] - curr[cidx][didx]).abs();
            }
        }
        delta
    }
    pub fn solve(&mut self, data:&[Vec<f32>]) {
        let mut prev = self.means.clone();
        let mut delta = 1_f32;
        while delta > CONVERGENCE_CONDITION {
            self.expectation_maximization(data);
            delta = self.delta(&prev, &self.means);
            prev = self.means.clone();
        }
    }
}
    // pub fn expectation_maximization(&mut self, data:&[Vec<f32>]) {
    //     let mut sum_linear = vec![vec![0_f32; self.cardinality]; self.centroids];
    //     let mut sum_squares = vec![generate_zero_matrix(self.cardinality, self.cardinality); self.centroids];
        
    //     let n = data.len();
    //     let mut x_bar = vec![0_f32; self.cardinality];
    //     let mut nweighted = vec![0_f32; self.centroids];
    //     let mut probs = vec![0_f32; self.centroids];
    //     let mut lus = Vec::with_capacity(self.centroids);
    //     let mut dets = Vec::with_capacity(self.centroids);
    //     let mut z_buf= vec![0_f32; self.cardinality];
    //     for k in 0..self.centroids {
    //         let lu = lu_decompose(self.variance[k].clone());
    //         dets.push(lu.find_determinant());
    //         lus.push(lu);
    //     }
    //     for x_i in data {
    //         let mut scaler = 0_f32;
    //         for k in 0..self.centroids {
    //             for c in 0..self.cardinality {
    //                 let val = x_i[c] - self.means[k][c];
    //                 x_bar[c] = val;
    //                 z_buf[c] = val;
    //             }
    //             probs[k] = self.mixtures[k] * gaussian(&mut x_bar, &mut z_buf, dets[k], &lus[k]);
    //             scaler += probs[k];
    //         }
    //         for k in 0..self.centroids {
    //             let pr = probs[k] / scaler;
    //             nweighted[k] += pr;
    //             for c in 0..self.cardinality {
    //                 sum_linear[k][c] += pr * x_i[c];
    //             }
    //             for i in 0..self.cardinality {
    //                 for j in 0..=i {
    //                     sum_squares[k].data[i * self.cardinality + j] += pr * x_i[i] * x_i[j] + EPSILON;
    //                 }
    //             }
    //         }
    //     }
    //     for k in 0..self.centroids {
    //         self.mixtures[k] = nweighted[k] / n as f32;
    //         for c in 0..self.cardinality {
    //             sum_linear[k][c] /= nweighted[k];
    //         }
    //         for i in 0..self.cardinality {
    //             for j in 0..=i {
    //                 sum_squares[k].data[i * self.cardinality + j] /= nweighted[k];
    //                 sum_squares[k].data[i * self.cardinality + j] -= sum_linear[k][i] * sum_linear[k][j];
    //                 sum_squares[k].data[j * self.cardinality + i] = sum_squares[k].data[i * self.cardinality + j];
    //             }
    //         }
    //     }
    //     self.means = sum_linear;
    //     self.variance = sum_squares;
    // }
