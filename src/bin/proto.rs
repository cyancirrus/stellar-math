use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::jacobi;
use stellar::structure::ndarray::NdArray;


fn main() {
    // {
        // Eigen values 2, -1
        let mut data = vec![0_f32; 4];
        let dims = vec![2; 2];
        data[0] = -1_f32;
        data[1] = 0_f32;
        data[2] = 5_f32;
        data[3] = 2_f32;
    // }
    // {
    //     data = vec![0_f32; 9];
    //     dims = vec![3; 2];
    //     data[0] = 1_f32;
    //     data[1] = 2_f32;
    //     data[2] = 3_f32;
    //     data[3] = 3_f32;
    //     data[4] = 4_f32;
    //     data[5] = 5_f32;
    //     data[6] = 6_f32;
    //     data[7] = 7_f32;
    //     data[8] = 8_f32;
    // }
    let x = NdArray::new(dims, data.clone());
    println!("x: {:?}", x);
    //
    let reference = golub_kahan_explicit(x.clone());
    println!("Reference {:?}", reference);
    
    let y = qr_decompose(x.clone());
    println!("triangle {:?}", y.triangle);
    
    let real_schur = real_schur(x.clone());
    println!("real schur kernel {:?}", real_schur.kernel);

}

