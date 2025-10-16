use rand::Rng;
use rand_distr::StandardNormal;
use plotters::prelude::*;

use rand::seq::SliceRandom;
use itertools::Itertools;
use stellar::learning::gaussian_mixture::{GaussianMixtureModel, kmeans_gmm_pipeline}; 
// TODO: implement the smarter sum for SSE via kahan summation
// TODO: implement smarter givens bulge chasing which only updates bidiagonals
// TODO: keep buffer for decision tree as it's reused a bit

fn sample_gaussian_diag(mean: &[f32], var_diag: &[f32], rng: &mut impl Rng) -> Vec<f32> {
    let d= mean.len();
    let mut sample = vec![0_f32;d];
    for i in 0..d {
        let z:f32 = rng.sample(StandardNormal);
        // u + s * z
        sample[i] += mean[i] + var_diag[i].sqrt() * z;
    }
    sample

}

fn generate_gmm_data(
    weights: &[f32],
    means: &[Vec<f32>],
    covs: &[Vec<f32>], // diagonal variances for now
    n: usize,
) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(n);

    let mut cumulative = vec![0.0; weights.len()];
    cumulative[0] = weights[0];
    // make a cdf
    for k in 1..weights.len() {
        cumulative[k] = cumulative[k - 1] + weights[k];
    }
    for _ in 0..n {
        let r: f32 = rng.random();
        let mut k = 0;
        while k + 1 < cumulative.len() && r > cumulative[k] {
            k += 1;
        }
        data.push(sample_gaussian_diag(&means[k], &covs[k], &mut rng));
    }
    data
}

// fn test_gmm_2d() {
//     // known parameters
//     let weights = vec![0.4, 0.6];
//     let means = vec![
//         vec![0.0, 0.0],
//         vec![3.0, 3.0],
//     ];
//     let covs = vec![
//         vec![0.5, 0.5],  // diagonal covariance
//         vec![0.8, 0.4],
//     ];
//     let data = generate_gmm_data(&weights, &means, &covs, 1200);

//     let mut gmm = GaussianMixtureModel::new(2, 2);
//     gmm.solve(&data);

//     println!("True means: {:?}", means);
//     println!("Fitted means: {:?}", gmm.means);
//     println!("Fitted variance: {:?}", gmm.variance);
//     println!("Mixtures: {:?}", gmm.mixtures);
//     let error = mean_error(&means, &gmm.means);
//     println!("Mean error {:?}", error);
// }

// fn test_gmm_3d() {
//     // known parameters
//     let weights = vec![0.3, 0.3, 0.4];
//     let means = vec![
//         vec![0.0, 0.0, -1.0],
//         vec![3.0, 2.0, 3.0],
//         vec![1.0, -1.0, 0.0],
//     ];
//     let covs = vec![
//         vec![0.5, 0.2, 0.2],  // diagonal covariance
//         vec![0.8, 0.4, 0.2],
//         vec![0.3, 0.3, 0.4],
//     ];
//     let data = generate_gmm_data(&weights, &means, &covs, 3_000);

//     let mut gmm = GaussianMixtureModel::new(3, 3);
//     gmm.solve(&data);

//     println!("True means: {:?}", means);
//     println!("Fitted means: {:?}", gmm.means);
//     println!("Fitted variance: {:?}", gmm.variance);
//     println!("Mixtures: {:?}", gmm.mixtures);
//     let error = mean_error(&means, &gmm.means);
//     println!("Mean error {:?}", error);
// }

// fn test_gmm_3d_kmeans_gmm() {
//     // known parameters
//     let weights = vec![0.3, 0.3, 0.4];
//     let means = vec![
//         vec![0.0, 0.0, -1.0],
//         vec![3.0, 2.0, 3.0],
//         vec![1.0, -1.0, 0.0],
//     ];
//     let covs = vec![
//         vec![0.5, 0.2, 0.2],  // diagonal covariance
//         vec![0.8, 0.4, 0.2],
//         vec![0.3, 0.3, 0.4],
//     ];
//     let data = generate_gmm_data(&weights, &means, &covs, 3_000);

//     let mut gmm = kmeans_gmm_pipeline(3, 3, &data);

//     println!("True means: {:?}", means);
//     println!("Fitted means: {:?}", gmm.means);
//     println!("Fitted variance: {:?}", gmm.variance);
//     println!("Mixtures: {:?}", gmm.mixtures);
//     let error = mean_error(&means, &gmm.means);
//     println!("Mean error {:?}", error);
// }

fn mean_error(true_means: &[Vec<f32>], fitted_means: &[Vec<f32>]) -> f32 {
    // find the distribution closest to fitted values
    let mut error = 0_f32;
    let clusters = true_means.len();
    let cardinality = true_means[0].len();
    for k in 0..clusters {
        let mut min_distance = f32::MAX;
        for m in 0..clusters {
            let mut distance = 0_f32;
            for i in 0..cardinality {
                distance += (true_means[k][i] - fitted_means[m][i]).abs();
            }
            if distance < min_distance {
                min_distance = distance;
            }
        }
        error += min_distance;
    }
    error / clusters as f32
}

fn permutation_mean_error(true_means: &[Vec<f32>], fitted_means: &[Vec<f32>]) -> f32 {
    // seems to underestimate unsure why going to leave it as is
    let k = true_means.len();
    (0..k)
        .permutations(k)
        .map(|perm| {
            perm.iter().enumerate().map(|(i,&j)| {
                true_means[i].iter().zip(&fitted_means[j])
                    .map(|(a,b)| (a-b).powi(2))
                    .sum::<f32>()
            }).sum::<f32>()
        })
        .min_by(|a,b| a.partial_cmp(b).unwrap())
        .unwrap()
        / (k as f32)
}


// /// Generate a 2D scatter plot showing true vs fitted cluster centers
// fn plot_gmm_2d(
//     data: Vec<Vec<f32>>,
//     true_means: &[Vec<f32>],
//     gmm_means: &[Vec<f32>],
//     kmeans_gmm_means: &[Vec<f32>],
//     filename: &str,
// ) -> Result<(), Box<dyn std::error::Error>> {
//     let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
//     root.fill(&WHITE)?;

//     let x_range = data.iter().map(|v| v[0]).fold(f32::INFINITY..f32::NEG_INFINITY, |acc, x| acc.start.min(x)..acc.end.max(x));
//     let y_range = data.iter().map(|v| v[1]).fold(f32::INFINITY..f32::NEG_INFINITY, |acc, y| acc.start.min(y)..acc.end.max(y));

//     let mut chart = ChartBuilder::on(&root)
//         .caption("GMM Cluster Comparison", ("sans-serif", 30))
//         .margin(20)
//         .x_label_area_size(40)
//         .y_label_area_size(40)
//         .build_cartesian_2d(x_range.start..x_range.end, y_range.start..y_range.end)?;

//     chart.configure_mesh().draw()?;

//     // Plot a random subset of points
//     let mut rng = rand::rng();
//     let mut sample_points = data.clone();
//     sample_points.shuffle(&mut rng);
//     sample_points.truncate(300);
//     // let sample_points: Vec<_> = data.choose_multiple(&mut rng, 300).collect();
//     chart.draw_series(
//         sample_points.iter().map(|p| Circle::new((p[0], p[1]), 3, BLUE.filled())),
//     )?;
//     // True means
//     chart.draw_series(
//         true_means.iter().map(|m| Circle::new((m[0], m[1]), 8, BLACK.filled())),
//     )?;

//     // Fitted GMM
//     chart.draw_series(
//         gmm_means.iter().map(|m| Cross::new((m[0], m[1]), 8, RED.filled())),
//     )?;

//     // KMeans+GMM
//     chart.draw_series(
//         kmeans_gmm_means.iter().map(|m| Cross::new((m[0], m[1]), 8, GREEN.filled())),
//     )?;


//     // Add legend
//     chart.configure_series_labels()
//         .background_style(&WHITE.mix(0.8))
//         .border_style(&BLACK)
//         .draw()?;

//     Ok(())
// }

/// Generate a 2D scatter plot showing true vs fitted cluster centers.
/// Note: this function plots the data, true_means and two fitted-mean sets;
/// you can use it to produce separate files for each fitted model by
/// passing the appropriate vectors or empty vecs when you don't want to draw them.
fn plot_gmm_2d(
    data: &Vec<Vec<f32>>,
    true_means: &[Vec<f32>],
    gmm_means: Option<&[Vec<f32>]>,
    kmeans_gmm_means: Option<&[Vec<f32>]>,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (900, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    // compute ranges
    let x_min = data.iter().map(|v| v[0]).fold(f32::INFINITY, |a, b| a.min(b));
    let x_max = data.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let y_min = data.iter().map(|v| v[1]).fold(f32::INFINITY, |a, b| a.min(b));
    let y_max = data.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, |a, b| a.max(b));

    // padding to avoid degenerate ranges
    let x_span = (x_max - x_min).abs().max(1e-3);
    let y_span = (y_max - y_min).abs().max(1e-3);
    let pad_x = x_span * 0.1;
    let pad_y = y_span * 0.1;

    let x_range = (x_min - pad_x)..(x_max + pad_x);
    let y_range = (y_min - pad_y)..(y_max + pad_y);

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("GMM Cluster Comparison — {}", filename), ("sans-serif", 24))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh().draw()?;

    // sample subset of points to plot
    let mut rng = rand::rng();
    let mut sample_points = data.clone();
    sample_points.shuffle(&mut rng);
    sample_points.truncate(500);

    // Data points
    chart
        .draw_series(
            sample_points
                .iter()
                .map(|p| Circle::new((p[0], p[1]), 2, BLUE.filled())),
        )?
        .label("Data (sample)")
        .legend(|(x, y)| Circle::new((x, y), 4, BLUE.filled()));

    // True means
    chart
        .draw_series(
            true_means
                .iter()
                .map(|m| Circle::new((m[0], m[1]), 8, BLUE.filled())),
        )?
        .label("True means")
        .legend(|(x, y)| Circle::new((x, y), 8, BLUE.filled()));

    // Raw GMM means (if provided)
    if let Some(gmm) = gmm_means {
        chart
            .draw_series(gmm.iter().map(|m| Cross::new((m[0], m[1]), 12, BLACK.filled())))?
            .label("GMM means")
            .legend(|(x, y)| Cross::new((x, y), 12, BLACK.filled()));
    }

    // KMeans+GMM means (if provided)
    if let Some(km) = kmeans_gmm_means {
        chart
            .draw_series(km.iter().map(|m| Cross::new((m[0], m[1]), 8, GREEN.filled())))?
            .label("KMeans→GMM means")
            .legend(|(x, y)| Cross::new((x, y), 8, GREEN.filled()));
    }

    // Draw legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    Ok(())
}

fn test_gmm_2d_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // True parameters
    let weights = vec![0.3, 0.3, 0.4];
    let means = vec![vec![0.0, 0.0], vec![3.0, 2.0], vec![1.0, -1.0]];
    let covs = vec![vec![0.5, 0.2], vec![0.8, 0.4], vec![0.3, 0.3]];

    // Generate shared data
    let data = generate_gmm_data(&weights, &means, &covs, 3_000);

    // Raw GMM
    let mut gmm = GaussianMixtureModel::new(3, 2);
    gmm.solve(&data);

    // KMeans + GMM
    let mut kmeans_gmm = kmeans_gmm_pipeline(3, 2, &data);

    // Print results
    println!("Raw GMM means: {:?}", gmm.means);
    println!("KMeans+GMM means: {:?}", kmeans_gmm.means);
    println!("Mean error (GMM): {:?}", mean_error(&means, &gmm.means));
    println!(
        "Mean error (KMeans+GMM): {:?}",
        mean_error(&means, &kmeans_gmm.means)
    );

    // Plot GMM-only
    plot_gmm_2d(&data, &means, Some(&gmm.means), None, "gmm_only.png")?;

    // Plot KMeans->GMM (separate PNG)
    plot_gmm_2d(&data, &means, None, Some(&kmeans_gmm.means), "kmeans_gmm.png")?;

    Ok(())
}

fn main() {
    if let Err(e) = test_gmm_2d_comparison() {
        eprintln!("error: {}", e);
    }
}


// /// Run a 2D comparison
// fn test_gmm_2d_comparison() -> Result<(), Box<dyn std::error::Error>> {
//     // True parameters
//     let weights = vec![0.3, 0.3, 0.4];
//     let means = vec![vec![0.0, 0.0], vec![3.0, 2.0], vec![1.0, -1.0]];
//     let covs = vec![vec![0.5, 0.2], vec![0.8, 0.4], vec![0.3, 0.3]];

//     // Generate shared data
//     let data = generate_gmm_data(&weights, &means, &covs, 3_000);

//     // Raw GMM
//     let mut gmm = GaussianMixtureModel::new(3, 2);
//     gmm.solve(&data);

//     // KMeans + GMM
//     let mut kmeans_gmm = kmeans_gmm_pipeline(3, 2, &data);

//     // Print results
//     println!("Raw GMM means: {:?}", gmm.means);
//     println!("KMeans+GMM means: {:?}", kmeans_gmm.means);
//     println!("Mean error (GMM): {:?}", mean_error(&means, &gmm.means));
//     println!("Mean error (KMeans+GMM): {:?}", mean_error(&means, &kmeans_gmm.means));

//     // Plot
//     plot_gmm_2d(data, &means, &gmm.means, &kmeans_gmm.means, "gmm_comparison.png")?;

//     Ok(())
// }


// fn main() {
//     // test_gmm_2d();
//     // test_gmm_3d();
//     // test_gmm_3d_kmeans_gmm();
// }
