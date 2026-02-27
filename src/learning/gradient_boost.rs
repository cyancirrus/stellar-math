use crate::learning::decision_tree::{DecisionTree, DecisionTreeModel};
pub struct GradientBoost {
    pub trees: usize,
    pub forest: Vec<DecisionTreeModel>,
}
impl GradientBoost {
    pub fn new(
        data: &mut Vec<Vec<f32>>,
        trees: usize,
        nodes: usize,
        obs_sample: f32,
        dim_sample: f32,
    ) -> Self {
        if data.is_empty() || data[0].is_empty() {
            panic!("data is empty");
        }
        let n_obs = data[0].len();
        let dims = data.len();
        let target_idx = data.len() - 1;
        let mut sample = vec![0_f32; dims];
        let mut forest = Vec::with_capacity(trees);
        for _ in 0..trees {
            let tree = DecisionTree::new(data, obs_sample, dim_sample).train(nodes);
            for idx in 0..n_obs {
                for d in 0..dims {
                    sample[d] = data[d][idx];
                }
                let pred = tree.predict(&sample);
                data[target_idx][idx] -= pred;
            }
            forest.push(tree);
        }
        Self { trees, forest }
    }
    pub fn predict(&self, data: &[f32]) -> f32 {
        let mut prediction = 0_f32;
        for tree in &self.forest {
            prediction += tree.predict(data);
        }
        prediction
    }
}
