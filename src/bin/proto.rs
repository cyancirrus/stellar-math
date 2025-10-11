#![allow(dead_code)]
use csv::ReaderBuilder;
use std::fs::File;
use stellar::learning::decision_tree::{DecisionTree, DecisionTreeModel};
use stellar::learning::gradient_boost::GradientBoost;
use stellar::learning::random_forest::RandomForest;

// TODO: implement the smarter sum for SSE via kahan summation
// TODO: implement smarter givens bulge chasing which only updates bidiagonals
// TODO: keep buffer for decision tree as it's reused a bit

fn main() {
}
