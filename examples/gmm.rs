use rand::Rng;
use rand_distr::StandardNormal;
use plotters::style::colors::full_palette::GREY_700;
use plotters::prelude::*;
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
    covs: &[Vec<f32>],
    n: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    // build cumulative CDF
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        cumulative.push(acc);
    }
    let total = *cumulative.last().unwrap_or(&1.0);
    for v in cumulative.iter_mut() {
        *v /= total;
    }

    for _ in 0..n {
        let r: f32 = rng.random();
        let mut k = 0;
        while k + 1 < cumulative.len() && r > cumulative[k] {
            k += 1;
        }
        data.push(sample_gaussian_diag(&means[k], &covs[k], &mut rng));
        labels.push(k);
    }

    (data, labels)
}

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

// fn permutation_mean_error(true_means: &[Vec<f32>], fitted_means: &[Vec<f32>]) -> f32 {
//     // seems to underestimate unsure why going to leave it as is
//     let k = true_means.len();
//     (0..k)
//         .permutations(k)
//         .map(|perm| {
//             perm.iter().enumerate().map(|(i,&j)| {
//                 true_means[i].iter().zip(&fitted_means[j])
//                     .map(|(a,b)| (a-b).powi(2))
//                     .sum::<f32>()
//             }).sum::<f32>()
//         })
//         .min_by(|a,b| a.partial_cmp(b).unwrap())
//         .unwrap()
//         / (k as f32)
// }

fn plot_gmm_clusters_2d(
    data: &Vec<Vec<f32>>,
    labels: &Vec<usize>,
    true_means: &[Vec<f32>],
    fitted_means: &[Vec<f32>],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (900, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    // compute ranges
    let x_min = data.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
    let x_max = data.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max);
    let y_min = data.iter().map(|v| v[1]).fold(f32::INFINITY, f32::min);
    let y_max = data.iter().map(|v| v[1]).fold(f32::NEG_INFINITY, f32::max);
    let pad_x = (x_max - x_min).abs() * 0.1;
    let pad_y = (y_max - y_min).abs() * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption("KMeans→GMM Clustering", ("sans-serif", 28))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d((x_min - pad_x)..(x_max + pad_x), (y_min - pad_y)..(y_max + pad_y))?;

    chart.configure_mesh().draw()?;

    // color palette
    let palette = vec![&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW, &GREEN];

    // draw data points by cluster
    for (k, &color) in palette.iter().enumerate() {
        let points: Vec<_> = data
            .iter()
            .zip(labels.iter())
            .filter(|(_, &label)| label == k)
            .map(|(p, _)| (p[0], p[1]))
            .collect();
        if !points.is_empty() {
            chart
                .draw_series(points.iter().map(|&(x, y)| Circle::new((x, y), 2, (*color).filled())))?
                .label(format!("Cluster {}", k))
                .legend(move |(x, y)| Circle::new((x, y), 5, (*color).filled()));
        }
    }

    // true means
    chart
        .draw_series(
            true_means
                .iter()
                .map(|m| TriangleMarker::new((m[0], m[1]), 10, GREY_700.filled())),
        )?
        .label("True means")
        .legend(|(x, y)| TriangleMarker::new((x, y), 8, GREY_700.filled()));

    // fitted means (KMeans→GMM)
    chart
        .draw_series(
            fitted_means
                .iter()
                .map(|m| Cross::new((m[0], m[1]), 8, BLACK.filled().stroke_width(2))),
        )?
        .label("Fitted means (KMeans→GMM)")
        .legend(|(x, y)| Cross::new((x, y), 8, BLACK.filled().stroke_width(2)));

    chart .configure_series_labels() .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn test_kmeans_gmm_visual() -> Result<(), Box<dyn std::error::Error>> {
    let weights = vec![0.3, 0.3, 0.4];
    let means = vec![vec![0.0, 0.0], vec![3.0, 2.0], vec![1.0, -1.0]];
    let covs = vec![vec![0.5, 0.2], vec![0.8, 0.4], vec![0.3, 0.3]];

    let (data, labels) = generate_gmm_data(&weights, &means, &covs, 3000);

    let mut gmm = kmeans_gmm_pipeline(3, 2, &data);

    println!("Fitted means (KMeans→GMM): {:?}", gmm.means);
    println!("Mean error: {:?}", mean_error(&means, &gmm.means));

    plot_gmm_clusters_2d(&data, &labels, &means, &gmm.means, "kmeans_gmm_clusters.png")?;
    Ok(())
}

fn main() {
    // test_gmm_3d_kmeans_gmm();
    // test_gmm_3d();
    if let Err(e) = test_kmeans_gmm_visual() {
        eprintln!("Error: {}", e);
    }
}
