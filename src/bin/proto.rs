#![allow(dead_code)]
use stellar::learning::decision_tree::{
    DecisionTree,
    DecisionTreeModel
}; 

use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
// #[cfg(target_arch = "x86_64")]

// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR

// move code into examples directory
// cargo run --example demo

// TODO: implement this, and then test return this from train, also add a like step through the
// vec<metadata> which computes the like increase in 1- total variance explained per step, for easy
// visualization, will help debugging 

fn read_boston_data() -> Vec<Vec<f32>> {
    let file = File::open("test_data/boston_housing.csv").unwrap();
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    // Read header to know number of columns
    let headers = rdr.headers().unwrap().clone();
    let n_cols = headers.len();

    // We'll collect columns (feature-major layout)
    let mut data: Vec<Vec<f32>> = vec![Vec::new(); n_cols];

    for result in rdr.records() {
        let record = result.unwrap();
        for (i, field) in record.iter().enumerate() {
            let val: f32 = field.parse().unwrap_or(f32::NAN);
            data[i].push(val);
        }
    }

    println!("Loaded {} columns × {} rows", n_cols, data[0].len());
    println!("First column {:?} -> {:?}", headers, &data[0][0..5.min(data[0].len())]);
    data
}

fn see_if_decision_tree_trains() -> DecisionTreeModel {
    let data = read_boston_data();
    let mut dt = DecisionTree::new(&data);
    let model = dt.train(8);
    let len = data.len();
    let mut test = vec![0_f32; len];
    let idx = 32;
    for i in 0..len {
        test[i] = data[i][idx];
    }
    println!("input_data {test:?}");
    let prediction = model.predict(&test);
    println!("prediction {prediction:?}");
    model
}



fn main() {
    let model = see_if_decision_tree_trains();
}
