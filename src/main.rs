use rand::Rng;
use rand_distr::StandardNormal;
use stellar::solver::multi_armed_bandit::{max_multi_bandit, Aggregate, Bandit};

fn sim() {
    // interesting that as t increased ratio -> 1;
    let n = 8;
    let t = 124;
    let sims = 4;
    let mut rng = rand::rng();
    for _ in 0..sims {
        let mut bandits = Vec::with_capacity(n);
        let mut aggs = Vec::with_capacity(n);
        println!("---------------------------");
        let mut best = f32::NEG_INFINITY;

        for _ in 0..n {
            let mean: f32 = rng.sample(StandardNormal);
            // let mut std_dev: f32 = rng.sample(StandardNormal);
            // std_dev = std_dev.abs();
            let std_dev = 0.5;
            best = best.max(mean);
            bandits.push(Bandit::new(mean, std_dev));
            aggs.push(Aggregate {
                sum: 0f32,
                square: 0f32,
                count: 0f32,
            });
        }
        let max = max_multi_bandit(&mut bandits, &mut aggs, t);
        let poss = best * t as f32;
        println!("achieved max {max:?}");
        println!("possible mean {poss:?}");
        println!("ratio {:?}", max / poss);
    }
}

fn main() {
    sim();
}
