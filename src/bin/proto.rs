use rand::Rng;
use rand_distr::StandardNormal;

use stellar::learning::expectation_maximization::GaussianMixtureModel; 
use stellar::learning::kmeans::Kmeans; 
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

fn mean_error(true_means: &[Vec<f32>], est_means: &[Vec<f32>]) -> f32 {
    let min_err = f32::MAX;
    // account for permutation
    let err_01 = (true_means[0][0] - est_means[0][0]).abs() +
                 (true_means[0][1] - est_means[0][1]).abs() +
                 (true_means[1][0] - est_means[1][0]).abs() +
                 (true_means[1][1] - est_means[1][1]).abs();
    let err_10 = (true_means[0][0] - est_means[1][0]).abs() +
                 (true_means[0][1] - est_means[1][1]).abs() +
                 (true_means[1][0] - est_means[0][0]).abs() +
                 (true_means[1][1] - est_means[0][1]).abs();
    min_err.min(err_01.min(err_10))
}


fn main() {
    // test_gmm_2d();
    test_gmm_3d();
}
