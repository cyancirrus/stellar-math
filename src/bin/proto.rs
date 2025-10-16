use rand::Rng;
use rand_distr::StandardNormal;

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

fn test_gmm_3d() {
    // known parameters
    let weights = vec![0.3, 0.3, 0.4];
    let means = vec![
        vec![0.0, 0.0, -1.0],
        vec![3.0, 2.0, 3.0],
        vec![1.0, -1.0, 0.0],
    ];
    let covs = vec![
        vec![0.5, 0.2, 0.2],  // diagonal covariance
        vec![0.8, 0.4, 0.2],
        vec![0.3, 0.3, 0.4],
    ];
    let data = generate_gmm_data(&weights, &means, &covs, 3_000);

    let mut gmm = GaussianMixtureModel::new(3, 3);
    gmm.solve(&data);

    println!("True means: {:?}", means);
    println!("Fitted means: {:?}", gmm.means);
    println!("Fitted variance: {:?}", gmm.variance);
    println!("Mixtures: {:?}", gmm.mixtures);
    let error = mean_error(&means, &gmm.means);
    println!("Mean error {:?}", error);
}

fn test_gmm_3d_kmeans_gmm() {
    // known parameters
    let weights = vec![0.3, 0.3, 0.4];
    let means = vec![
        vec![0.0, 0.0, -1.0],
        vec![3.0, 2.0, 3.0],
        vec![1.0, -1.0, 0.0],
    ];
    let covs = vec![
        vec![0.5, 0.2, 0.2],  // diagonal covariance
        vec![0.8, 0.4, 0.2],
        vec![0.3, 0.3, 0.4],
    ];
    let data = generate_gmm_data(&weights, &means, &covs, 3_000);

    let mut gmm = kmeans_gmm_pipeline(3, 3, &data);

    println!("True means: {:?}", means);
    println!("Fitted means: {:?}", gmm.means);
    println!("Fitted variance: {:?}", gmm.variance);
    println!("Mixtures: {:?}", gmm.mixtures);
    let error = mean_error(&means, &gmm.means);
    println!("Mean error {:?}", error);
}

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



fn main() {
    // test_gmm_2d();
    test_gmm_3d();
test_gmm_3d_kmeans_gmm();
}
