use rand::Rng;
use rand_distr::StandardNormal;
use stellar::solver::multi_armed_bandit::{max_multi_bandit, Aggregate, Bandit};

fn sim() {
    // interesting that as t increased ratio -> 1;
    let n = 8;
    let t = 128;
    let sims = 12;
    let mut rng = rand::rng();
    let mut bandits = Vec::with_capacity(n);
    let mut aggs = Vec::with_capacity(n);
    for _ in 0..sims {
        println!("---------------------------");
        let mut best = f32::NEG_INFINITY;

        for _ in 0..n {
            let mean: f32 = rng.sample(StandardNormal);
            let std_dev: f32 = rng.sample(StandardNormal);
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
