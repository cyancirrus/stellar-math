use stellar::learning::knn::LshKNearestNeighbors;

// #[cfg(target_arch = "x86_64")]
use rand::Rng;
use rand_distr::StandardNormal;
use rand::distr::StandardUniform;
use rand::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;

fn generate_clusters(num_points: usize, dim: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let mut data = Vec::new();
    
    // random cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-10.0..10.0) as f32).collect())
        .collect();

    let normal = Normal::new(0.0, 1.0).unwrap();
    
    for _ in 0..num_points {
        // pick a random cluster
        let c = &centers[rng.random_range(0..num_clusters)];
        // sample around center
        let point: Vec<f32> = c.iter()
            .map(|&v| v + normal.sample(&mut rng) as f32)
            .collect();
        data.push(point);
    }
    
    data
}

fn main() {
    let data = generate_clusters(100, 2, 3); // 100 points, 2D, 3 clusters
    // for p in &data {
    //     println!("{:?}", p);
    // }
    let mut knn = LshKNearestNeighbors::new(7, 2, 6); 
    knn.parse(data.clone());
    // for p in &data {
    //     println!("{:?}", p);
    // }
    let result = knn.knn(5, data[0].clone());
    println!("--------------");
    for p in &result {
        println!("{:?}", p);
    }
}


// fn main() {
//     let mut rng = rand::rng();
//     let x:f32 = rng.sample(StandardNormal);
//     println!("result {x:?}");
// }

