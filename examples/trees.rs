#![allow(dead_code)]
use rand::seq::SliceRandom;
use stellar::learning::decision_tree::{DecisionTree, DecisionTreeModel};
use stellar::learning::gradient_boost::GradientBoost;
use stellar::learning::random_forest::RandomForest;

use csv::ReaderBuilder;
use plotters::prelude::*;
use std::fs::File;

fn read_boston_data() -> Vec<Vec<f32>> {
    let file = File::open("test_data/boston_housing.csv").unwrap();
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    // read header to know number of columns
    let headers = rdr.headers().unwrap().clone();
    let n_cols = headers.len();
    // feature-major layout
    let mut data: Vec<Vec<f32>> = vec![Vec::new(); n_cols];

    for result in rdr.records() {
        let record = result.unwrap();
        for (i, field) in record.iter().enumerate() {
            let val: f32 = field.parse().unwrap_or(f32::NAN);
            data[i].push(val);
        }
    }
    data
}

fn plot_results(results: &Vec<Vec<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("tve_chart.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("TVE Comparison", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..results[0].len(), 0f32..1f32)?;

    chart.configure_mesh().draw()?;

    let series_labels = ["Decision Tree", "Random Forest", "Gradient Boost"];
    let colors = [&RED, &BLUE, &GREEN];

    for (i, series) in results.iter().enumerate() {
        chart
            .draw_series(LineSeries::new(
                series.iter().enumerate().map(|(x, y)| (x, *y)),
                colors[i],
            ))?
            .label(series_labels[i])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i]));
    }

    // configure legend style and draw it
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    Ok(())
}

// fn plot_results(results: &Vec<Vec<f32>>) -> Result<(), Box<dyn std::error::Error>> {
//     let root = BitMapBackend::new("tve_chart.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE)?;

//     let mut chart = ChartBuilder::on(&root)
//         .caption("TVE Comparison", ("sans-serif", 30))
//         .margin(20)
//         .x_label_area_size(40)
//         .y_label_area_size(40)
//         .build_cartesian_2d(0..results[0].len(), 0f32..1f32)?;

//     chart.configure_mesh().draw()?;

//     let colors = &[&RED, &BLUE, &GREEN];
//     for (i, series) in results.iter().enumerate() {
//         chart.draw_series(LineSeries::new(
//             series.iter().enumerate().map(|(x, y)| (x, *y)),
//             colors[i],
//         ))?;
//     }
//     Ok(())
// }

/// Compute Total Variance Explained (TVE) on test set
fn tve(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let mean: f32 = y_true.iter().sum::<f32>() / y_true.len() as f32;
    let ss_total: f32 = y_true.iter().map(|v| (v - mean).powi(2)).sum();
    let ss_res: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum();
    1.0 - ss_res / ss_total
}

/// Evaluate DecisionTree, RandomForest, GradientBoost on Boston data
fn evaluate_models(data: &Vec<Vec<f32>>) -> (f32, f32, f32) {
    let n_obs = data[0].len();
    let n_dims = data.len();
    let mut indices: Vec<usize> = (0..n_obs).collect();
    let mut rng = rand::rng();
    indices.shuffle(&mut rng);

    let split = (0.75 * n_obs as f32) as usize;
    let train_idx = &indices[..split];
    let test_idx = &indices[split..];

    // Prepare train set (feature-major, observations as columns)
    let train_data: Vec<Vec<f32>> = data
        .iter()
        .map(|col| train_idx.iter().map(|&i| col[i]).collect())
        .collect();

    // True labels for test set
    let target_idx = n_dims - 1;
    let y_test: Vec<f32> = test_idx.iter().map(|&i| data[target_idx][i]).collect();

    // ---------------- Decision Tree ----------------
    let mut dtree = DecisionTree::new(&train_data, 0.8, 0.8);
    let dtree_model = dtree.train(8);
    let y_pred_tree: Vec<f32> = test_idx
        .iter()
        .map(|&i| {
            let sample: Vec<f32> = (0..n_dims).map(|d| data[d][i]).collect();
            dtree_model.predict(&sample)
        })
        .collect();
    let tve_dtree = tve(&y_test, &y_pred_tree);
    println!("Decision Tree TVE: {:.2}%", 100.0 * tve_dtree);

    // ---------------- Random Forest ----------------
    let forest = RandomForest::new(&train_data, 50, 8, 0.8, 0.8);
    let y_pred_rf: Vec<f32> = test_idx
        .iter()
        .map(|&i| {
            let sample: Vec<f32> = (0..n_dims).map(|d| data[d][i]).collect();
            forest.predict(&sample)
        })
        .collect();

    let tve_forest = tve(&y_test, &y_pred_rf);
    println!("Random Forest TVE: {:.2}%", 100.0 * tve_forest);

    // ---------------- Gradient Boost ----------------
    let mut train_clone = train_data.clone();
    let gb = GradientBoost::new(&mut train_clone, 2, 6, 0.9, 0.9);
    let y_pred_gb: Vec<f32> = test_idx
        .iter()
        .map(|&i| {
            let sample: Vec<f32> = (0..n_dims).map(|d| data[d][i]).collect();
            gb.predict(&sample)
        })
        .collect();
    let tve_boost = tve(&y_test, &y_pred_gb);
    println!("Gradient Boost TVE: {:.2}%", 100.0 * tve_boost);
    (tve_dtree, tve_forest, tve_boost)
}

fn main() {
    let data = read_boston_data();
    let n = 36;
    let mut results = vec![vec![]; 3];
    for _ in 0..n {
        let (dtree, forest, boost) = evaluate_models(&data);
        results[0].push(dtree);
        results[1].push(forest);
        results[2].push(boost);
    }
    for i in 0..3 {
        results[i].sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let output = plot_results(&results);
}
