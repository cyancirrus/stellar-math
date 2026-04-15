use crate::sharedvars::{L_MATRIX_DIMS, M_MATRIX_DIMS, S_MATRIX_DIMS};
use pprof::ProfilerGuard;
use pprof::ProfilerGuardBuilder;
use stellar::algebra::mmethods::tensor_kernel_new;
use stellar::random::generation::generate_random_matrix;
use std::fs::File;
use std::hint::black_box;

const ITERS_PER_DIM:usize = 200;

pub fn run_flame() {
    // let guard = ProfilerGuard::new(100).expect("could not start profiler");
    let guard = ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["librc", "libgcc", "pthread", "vDSP"])
        .build()
        .expect("could not start profilier");

    for &(i, k, j) in &L_MATRIX_DIMS {
        let x = generate_random_matrix(i, k);
        let y = generate_random_matrix(k, j);
        let mut target = vec![0.0f32; i * j];

        for _ in 0..ITERS_PER_DIM {
            black_box(tensor_kernel_new(&x, &y, &mut target));
        }
    }

    match guard.report().build() {
        Ok(report) => {
            let path = "./diagnostics/flamegraph_matmul.svg";
            let file = File::create(path).expect("could not create file");
            report.flamegraph(file).expect("could not write file");
            println!("Flamegraph written :: {path:?}");
            println!("Open with firefox :: open -a \"Firefox\" {path:?}");
        },
        Err(e) => {
            println!("Failure in building report with error {e:?}");
        }
        
    }
}
