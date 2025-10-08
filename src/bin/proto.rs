#![allow(dead_code)]
use stellar::learning::decision_tree::{
    DecisionTree,
    DecisionTreeModel
}; 
// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR

// move code into examples directory
// cargo run --example demo

// TODO: implement this, and then test return this from train, also add a like step through the
// vec<metadata> which computes the like increase in 1- total variance explained per step, for easy
// visualization, will help debugging 

struct RandomForest {
    trees: usize,
    forest: Vec<DecisionTreeModel>,
}

impl RandomForest {
    fn new(data:&Vec<Vec<f32>>, trees:usize, nodes:usize, obs_sample:f32, dim_sample:f32) -> Self {
        let forest:Vec<DecisionTreeModel> = (0..trees).into_iter().map(|_| {
            let mut tree = DecisionTree::new(data, obs_sample, dim_sample);
            tree.train(nodes)
        }).collect();
        Self { trees, forest }
    }
    fn predict(&self, data:&[f32]) -> f32 {
        let mut cumulative = 0_f32;
        for tree in &self.forest {
            cumulative += tree.predict(data);
        }
        cumulative / self.trees as f32
    }
}


fn main() {
}
