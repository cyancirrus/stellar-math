use crate::algebra::vector::{dot_product, distance_squared,};
use rand::Rng;
use rand_distr::StandardNormal;
use rand::distr::StandardUniform;
use rand::prelude::*;
use std::collections::HashMap;

struct RandomProjection {
    // normally distributed
    a:Vec<f32>,
    // uniformly distributed
    b:f32,
    // bucket width
    w:f32,
}

impl RandomProjection {
    fn new(w:usize, n:usize) -> Self {
        let mut rng = rand::rng();
        Self {
            a: (0..n).map(|_| rng.sample(StandardNormal)).collect(),
            b:rng.sample::<f32, _>(StandardUniform) * w as f32,
            w:w as f32,
        }
    }
    fn project(&self, x:&[f32]) -> i32 {
        (( dot_product(&self.a, x) +  self.b ) / self.w ).floor() as i32
    }
}
type ProjHash = HashMap<i32, Vec<Vec<f32>>>;

struct LshKNearestNeighbors {
    w:usize, n:usize, h:usize,
    // local sensitivity hashing
    proj:Vec<RandomProjection>,
    pmaps:Vec<ProjHash>

}

impl LshKNearestNeighbors {
    fn new(w:usize, n:usize, h:usize) -> Self {
        debug_assert!(w > 0 && n > 0 && h > 0);
        Self {
            w, n, h,
            proj: (0..h).map(|_| RandomProjection::new(w, n)).collect(),
            pmaps: vec![HashMap::new();h],
        }
    }
    fn insert(&mut self, x:Vec<f32>) {
        debug_assert!(x.len() == self.proj[0].a.len());
        for h in 0..self.h {
            let hash = self.proj[h].project(&x);
            (self.pmaps[h].entry(hash).or_default()).push(x.clone())
        }
    }
    fn parse(&mut self, data:Vec<Vec<f32>>) {
        for d in data {
            self.insert(d);
        }
    }
    fn knn(&self, k:usize, x:Vec<f32>) -> Vec<Vec<f32>> {
        let mut similar = Vec::new();
        for i in 0..self.h {
            similar.extend(self.pmaps[i][&self.proj[i].project(&x)].clone());
        }
        similar.sort_by(|a, b| {
            let dist_a = distance_squared(a, &x);
            let dist_b = distance_squared(b, &x);
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        similar.truncate(k);
        similar
    }
}
