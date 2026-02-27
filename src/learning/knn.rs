use crate::algebra::vector::{distance_squared, dot_product};
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::collections::HashMap;
use std::sync::Arc;

struct RandomProjection {
    // normally distributed
    a: Vec<f32>,
    // uniformly distributed
    b: f32,
    // bucket width
    w: f32,
}

impl RandomProjection {
    fn new(w: usize, n: usize) -> Self {
        let mut rng = rand::rng();
        Self {
            a: (0..n).map(|_| rng.sample(StandardNormal)).collect(),
            b: rng.sample::<f32, _>(StandardUniform) * w as f32,
            w: w as f32,
        }
    }
    fn project(&self, x: &[f32]) -> i32 {
        ((dot_product(&self.a, x) + self.b) / self.w).floor() as i32
    }
}
type ProjHash = HashMap<i32, Vec<Arc<Vec<f32>>>>;

pub struct LshKNearestNeighbors {
    pub w: usize,
    pub n: usize,
    pub h: usize,
    // local sensitivity hashing
    proj: Vec<RandomProjection>,
    pmaps: Vec<ProjHash>,
}

impl LshKNearestNeighbors {
    pub fn new(w: usize, n: usize, h: usize) -> Self {
        debug_assert!(w > 0 && n > 0 && h > 0);
        Self {
            w,
            n,
            h,
            proj: (0..h).map(|_| RandomProjection::new(w, n)).collect(),
            pmaps: vec![HashMap::new(); h],
        }
    }
    pub fn insert(&mut self, x: Vec<f32>) {
        debug_assert!(x.len() == self.proj[0].a.len());
        let x_arc = Arc::new(x);
        for h in 0..self.h {
            let hash = self.proj[h].project(&x_arc);
            (self.pmaps[h].entry(hash).or_default()).push(x_arc.clone())
        }
    }
    pub fn parse(&mut self, data: Vec<Vec<f32>>) {
        for d in data {
            self.insert(d);
        }
    }
    pub fn knn(&self, k: usize, x: Vec<f32>) -> Vec<Arc<Vec<f32>>> {
        let mut similar = Vec::new();
        for i in 0..self.h {
            similar.extend(self.pmaps[i][&self.proj[i].project(&x)].clone());
        }
        similar.sort_by(|a, b| {
            let dist_a = distance_squared(a, &x);
            let dist_b = distance_squared(b, &x);
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        similar.dedup();
        similar.truncate(k);
        similar
    }
}

// TODO: Clean up into actual tests
// TODO: Sample code

// use rand::Rng;
// use rand_distr::StandardNormal;
// use rand::distr::StandardUniform;
// use rand::prelude::*;
// use rand::prelude::*;
// use rand_distr::Normal;

// fn generate_clusters(num_points: usize, dim: usize, num_clusters: usize) -> Vec<Vec<f32>> {
//     let mut rng = rand::rng();
//     let mut data = Vec::new();

//     // random cluster centers
//     let centers: Vec<Vec<f32>> = (0..num_clusters)
//         .map(|_| (0..dim).map(|_| rng.random_range(-10.0..10.0) as f32).collect())
//         .collect();

//     let normal = Normal::new(0.0, 1.0).unwrap();

//     for _ in 0..num_points {
//         // pick a random cluster
//         let c = &centers[rng.random_range(0..num_clusters)];
//         // sample around center
//         let point: Vec<f32> = c.iter()
//             .map(|&v| v + normal.sample(&mut rng) as f32)
//             .collect();
//         data.push(point);
//     }

//     data
// }

// fn main() {
//     let data = generate_clusters(100, 2, 3); // 100 points, 2D, 3 clusters
//     // for p in &data {
//     //     println!("{:?}", p);
//     // }
//     let mut knn = LshKNearestNeighbors::new(7, 2, 6);
//     knn.parse(data.clone());
//     // for p in &data {
//     //     println!("{:?}", p);
//     // }
//     let result = knn.knn(5, data[0].clone());
//     println!("--------------");
//     for p in &result {
//         println!("{:?}", p);
//     }
// }
