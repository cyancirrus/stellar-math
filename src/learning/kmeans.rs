use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::rngs::ThreadRng;

const CONVERGENCE_CONDITION: f32 = 1e-6;

// TODO: Note for the distance measure can reuse
// (x-u)'(x-u) = <x,x> + <u,u> - 2<u, x>;

pub struct Kmeans {
    centroids: usize,
    cardinality: usize,
    pub means: Vec<Vec<f32>>,
    pub mixtures: Vec<f32>,
}

fn initialize_distribution(n: usize, rng: &mut ThreadRng) -> Vec<f32> {
    (0..n).map(|_| rng.sample(StandardUniform)).collect()
}

fn squared_distance(x: &[f32], z: &[f32]) -> f32 {
    // assymptotically equal when S := I
    let mut squares = 0_f32;
    for i in 0..x.len() {
        squares += (x[i] - z[i]) * (x[i] - z[i]);
    }
    squares
}

impl Kmeans {
    pub fn new(centroids: usize, cardinality: usize) -> Self {
        let mut rng = rand::rng();
        let means = (0..centroids)
            .map(|_| initialize_distribution(cardinality, &mut rng))
            .collect();
        let mixtures = vec![1_f32 / centroids as f32; centroids];
        Self {
            centroids,
            cardinality,
            means,
            mixtures,
        }
    }
    fn maximization(&mut self, data: &[Vec<f32>]) {
        let mut sum_linear = vec![vec![0_f32; self.cardinality]; self.centroids];
        let mut cluster_ns = vec![0; self.centroids];
        let n = data.len();
        for i in 0..n {
            let mut min_dist = f32::MAX;
            let mut min_k = 0;
            for k in 0..self.centroids {
                let ln_prob = squared_distance(&data[i], &self.means[k]);
                if ln_prob < min_dist {
                    min_dist = ln_prob;
                    min_k = k;
                }
            }
            for k in 0..self.cardinality {
                sum_linear[min_k][k] += data[i][k];
            }
            cluster_ns[min_k] += 1;
        }
        for k in 0..self.centroids {
            for c in 0..self.cardinality {
                sum_linear[k][c] /= cluster_ns[k].max(1) as f32;
            }
            self.mixtures[k] = cluster_ns[k] as f32 / n as f32;
        }
        self.means = sum_linear;
    }
    fn delta(&self, prev: &[Vec<f32>], curr: &[Vec<f32>]) -> f32 {
        let mut delta = 0_f32;
        for cidx in 0..self.centroids {
            for didx in 0..self.cardinality {
                delta += (prev[cidx][didx] - curr[cidx][didx]).abs();
            }
        }
        delta
    }
    pub fn solve(&mut self, data: &[Vec<f32>]) {
        let mut prev = self.means.clone();
        let mut delta = 1_f32;
        while delta > CONVERGENCE_CONDITION {
            self.maximization(data);
            delta = self.delta(&prev, &self.means);
            prev = self.means.clone();
        }
    }
}
