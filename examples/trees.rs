#![allow(dead_code)]
use stellar::learning::decision_tree::{
    DecisionTree,
    DecisionTreeModel
};
use stellar::learning::gradient_boost::GradientBoost;
use stellar::learning::random_forest::RandomForest;
// reading : https://en.wikipedia.org/wiki/Schur_complement
// when writing article write something about recursive / iterative defns and which info is
// available, ie why need to reverse iteration on QR

// move code into examples directory
// cargo run --example demo

// TODO: implement this, and then test return this from train, also add a like step through the
// vec<metadata> which computes the like increase in 1- total variance explained per step, for easy
// visualization, will help debugging 
// #![allow(dead_code)]

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

    // println!("Loaded {} columns × {} rows", n_cols, data[0].len());
    // println!("First column {:?} -> {:?}", headers, &data[0][0..5.min(data[0].len())]);
    data
}

fn checking_decision_model() -> DecisionTreeModel {
    let data = read_boston_data();
    let mut dt = DecisionTree::new(&data, 0.8, 0.8);
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

    println!("analyzing variance");
    model.analyze_variance();
    model
}


fn checking_forest_model() -> RandomForest {
    println!("Random Forest");
    let data = read_boston_data();
    let forest = RandomForest::new(&data, 124, 10, 0.8, 0.8);
    let len = data.len();
    let mut test = vec![0_f32; len];
    let idx = 32;
    for i in 0..len {
        test[i] = data[i][idx];
    }
    println!("input_data {test:?}");
    let prediction = forest.predict(&test);
    println!("prediction {prediction:?}");
    forest
}

fn checking_gradient_model() -> GradientBoost {
    println!("Gradient Boosting");
    let data = read_boston_data();
    let mut data_train = data.clone();
    let gradient = GradientBoost::new(&mut data_train, 3, 6, 0.9, 0.9);
    let len = data.len();
    let mut test = vec![0_f32; len];
    let idx = 32;
    for i in 0..len {
        test[i] = data[i][idx];
    }
    println!("input_data {test:?}");
    let prediction = gradient.predict(&test);
    println!("prediction {prediction:?}");
    gradient
}


fn main() {
    checking_decision_model();
    println!("--------------------------------------");
    checking_forest_model();
    println!("--------------------------------------");
    checking_gradient_model();
    println!("--------------------------------------");
}
