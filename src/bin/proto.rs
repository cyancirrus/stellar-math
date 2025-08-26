use stellar::decomposition::svd::golub_kahan_explicit;
use stellar::decomposition::schur::real_schur;
use stellar::decomposition::qr::qr_decompose;
use stellar::decomposition::givens::givens_iteration;
use stellar::structure::ndarray::NdArray;
use stellar::decomposition::householder::householder_factor;

// Av = lambda v
// (A - lambda I)v = 0
// use householder in order to reduce A to upper triangle

const EPSILON:f32 = 1e-6;

fn scale_identity(n:usize, c:f32) -> NdArray {
    let mut mat = vec![0_f32;n*n];
    for i in 0..n {
        mat[i * n + i] = c;
    }
    NdArray::new(vec![2;2], mat)
}

fn target_rotation_indices(matrix:&NdArray) -> (usize, usize) {
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let m = matrix.dims[0];
    let mut rswap = m-1;
    for i in 0..m {
        if matrix.data[i * m + i].abs() < EPSILON {
            rswap = i;
            break;
        }
    }
    return (rswap, m-1)
}

fn row_swap(i:usize, j:usize, matrix:&mut NdArray) {
    debug_assert!(matrix.dims.len() == 2);
    debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let m = matrix.dims[1];
    if i < usize::MAX && j < usize::MAX {
        for k in 0..m {
            matrix.data.swap(i * m + k, j * m + k);
        }
    }
}

fn normalize(v:&mut Vec<f32>) {
    let mut norm = 0_f32;
    for i in 0..v.len() {
        norm += v[i] * v[i];
    }
    norm = norm.sqrt();
    for i in 0..v.len() {
        v[i] /= norm;
    }
}

fn retrieve_eigen(eig:f32, mut matrix:NdArray) -> Vec<f32> {
    // debug_assert!(matrix.dims.len() == 2);
    // debug_assert!(matrix.dims[0] == matrix.dims[1]);
    let m = matrix.dims[0];
    let mut evector = vec![0_f32;m];
    let lambda_i = scale_identity(m, eig);
    // (A - lambda I)v = 0
    // (A - lambda I) = L
    // Lv = 0
    matrix.diff(lambda_i);
    // // B is a rotation matrix
    // // B L v = 0;
    let (i, j) = target_rotation_indices(&matrix);
    println!("i:{i:}, j:{j:} \n{matrix:?}");
    row_swap(i, j, &mut matrix);
    println!("matrix {matrix:?}");
    // // QR = (B L )
    // // QRv = 0
    // // Q'Qrv = 0
    let qr = householder_factor(matrix);
    println!("Q {:?}", qr.projections);
    println!("Triangle {:?}", qr.triangle);
    // // Rv' = 0 <-> O'v :: QRv = 0
    for i in (0..m).rev() {
        let diag = qr.triangle.data[i * m + i];
        if diag.abs() < EPSILON {
            evector[i] = 1_f32;
        } else {
            let mut sum = 0_f32;
            for j in i+1..m {
                sum += qr.triangle.data[i * m + j] * evector[j];
            }
            evector[i] = -sum / diag;
        }
    }
    println!("evector {evector:?}");
    evector.swap(i, j);
    normalize(&mut evector);
    evector
}


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
    // eigenvalues
    println!("real schur kernel {:?}", real_schur.kernel);

    let svd = givens_iteration(reference);
    println!("svd u, s, v \nU: {:?}, \nS: {:?}, \nV: {:?}",svd.u, svd.s, svd.v);

    let evector = retrieve_eigen(real_schur.kernel.data[3], x.clone());
    println!("eigen vec {evector:?}");


}

