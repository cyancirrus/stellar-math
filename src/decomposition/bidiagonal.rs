use crate::decomposition::givens::implicit_givens_rotation;
use crate::structure::ndarray::NdArray;

// https://www.netlib.org/lapack/lawnspdf/lawn03.pdf
// https://www.cs.utexas.edu/~flame/pubs/RestructuredQRTOMS.pdf
pub fn bidiagonal_qr(mut b: NdArray) -> NdArray {
    // b :: Bidiagonalized Matrix
    let rows = b.dims[0];
    let cols = b.dims[1];
    assert!(rows > 1, "Have not handled trivial cases");
    assert!(cols > 1, "Have not handled trivial cases");
    assert_eq!(rows, cols, "Sigma should be a square matrix in all cases");

    // let cosine = 1_f32;
    // let sine= 0_f32;
    let mut sigma = b.data[0];
    let mut supra;
    let mut cosine: f32 = 1_f32;
    let mut sine: f32 = 0_f32;
    let mut h: f32 = b.data[1];
    for i in 0..rows - 1 {
        supra = b.data[i * cols + i + 1];
        let (r, c, s) = implicit_givens_rotation(sigma, supra);
        if i != 0 {
            b.data[(i - 1) * cols + i] = sine * r;
        }
        sigma = cosine * r;
        supra = b.data[(i + 1) * cols + (i + 1)] * s;
        h = b.data[(i + 1) * cols + (i + 1)] * c;
        println!("sigma {}, supra{}", sigma, supra);
        let (r, c, s) = implicit_givens_rotation(sigma, supra);
        println!("hello i am did implicit print?");
        b.data[i * cols + i] = r;
        sigma = h;
        // println!("Check data {:?}, supra {}", b, supra);
        cosine = c;
        sine = s;
        println!("sine {}, cosine {}", s, c,);
    }
    b.data[(rows - 1) * cols - 1] = h * sine;
    b.data[rows * cols - 1] = h * cosine;
    b
}

pub fn fast_bidiagonal_qr(mut b: NdArray) -> NdArray {
    // b :: Bidiagonalized Matrix returns the sigma matrix
    let rows = b.dims[0];
    let cols = b.dims[1];
    assert!(rows > 1, "Have not handled trivial cases");
    assert!(cols > 1, "Have not handled trivial cases");
    assert_eq!(rows, cols, "Sigma should be a square matrix in all cases");

    let mut cosine: f32 = 1_f32;
    let mut sigma: f32 = b.data[0];
    let mut sine: f32;
    let mut supra: f32;
    let mut old_cosine: f32 = 1_f32;
    let mut old_sine: f32 = 1_f32;
    for i in 0..rows - 1 {
        // sigma = b.data[i * cols + i];
        supra = b.data[i * cols + i + 1];
        let (r, c, s) = implicit_givens_rotation(sigma * cosine, supra);
        cosine = c;
        sine = s;
        if i != 0 {
            b.data[(i - 1) * cols + i] = sine * r;
        }
        sigma = b.data[(i + 1) * cols + (i + 1)];
        let (r, c, s) = implicit_givens_rotation(old_cosine * r, sigma * sine);
        sigma = r;
        old_cosine = c;
        old_sine = s;
        b.data[i * cols + i] = r;
    }
    let scale = b.data[cols * rows - 1] * cosine;
    b.data[(rows - 1) * cols - 1] = scale * old_sine;
    b.data[rows * cols - 1] = scale * old_cosine;
    b.data[rows * cols - 1] = scale * old_cosine;
    b
}
