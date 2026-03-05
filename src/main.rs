use stellar::solver::linear::LinearProgram;
use stellar::structure::ndarray::NdArray;

fn main() {
    // let thing = vec![vec![3], vec![1,3], vec![2], vec![2,3], vec![0,2], vec![0,1]];
    let c = vec![1.0; 4];
    let b = vec![3.0, 5.0, 4.0, 7.0];
    let matrix = NdArray::new(
        vec![4, 4],
        vec![
            1.0, 0.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
        ],
    );
    println!("matrix {matrix:?}");
    let mut lp = LinearProgram::new(c, b, matrix);
    println!("phase one");
    let res_one = lp.run_phase_one();
    println!("phase two");
    let res_two = lp.run_phase_two();
    match (res_one, res_two) {
        (Ok(r1), Ok(r2)) => {
            println!("Successful run of both!");
            println!("Result {r2:?}");
        }
        (Err(e1), Err(e2)) => {
            println!("Unsucessful\ne1: {e1:?}\ne2: {e2:?}");
        }
        (Err(e1), Ok(_)) => {
            println!("Unsucessful\ne1: {e1:?}");
        }
        (Ok(_), Err(e2)) => {
            println!("Unsucessful\ne2: {e2:?}");
        }
    }
    // // test_gmm_3d_kmeans_gmm();
    // // test_gmm_3d();
    // if let Err(e) = test_kmeans_gmm_visual() {
    //     eprintln!("Error: {}", e);
    // }
}
