use crate::learning::decision_tree::{DecisionTree, DecisionTreeModel};
pub struct RandomForest {
    trees: usize,
    forest: Vec<DecisionTreeModel>,
}
impl RandomForest {
    pub fn new(
        data: &Vec<Vec<f32>>,
        trees: usize,
        nodes: usize,
        obs_sample: f32,
        dim_sample: f32,
    ) -> Self {
        let forest: Vec<DecisionTreeModel> = (0..trees)
            .into_iter()
            .map(|_| {
                let mut tree = DecisionTree::new(data, obs_sample, dim_sample);
                tree.train(nodes)
            })
            .collect();
        Self { trees, forest }
    }
    pub fn predict(&self, data: &[f32]) -> f32 {
        let mut cumulative = 0_f32;
        for tree in &self.forest {
            cumulative += tree.predict(data);
        }
        cumulative / self.trees as f32
    }
}
