#![allow(unused)]
use std::time::Instant;
use stellar::algebra::bmethods::contractions::{
    tensor_lt_contraction, tensor_rut_contraction, tensor_tlt_contraction, tensor_tut_contraction,
    tensor_ut_contraction,
};
use stellar::algebra::bmethods::interface::{tensor_kernel, tensor_tlt_kernel, tensor_tut_kernel};
use stellar::algebra::ndmethods::create_identity_matrix;
use stellar::algebra::ndmethods::{create_identity_vector, matrix_mult};
use stellar::decomposition::lq::AutumnDecomp;
use stellar::decomposition::wy::wy_decomposition;
use stellar::random::generation::generate_random_vector;
use stellar::structure::ndarray::NdArray;

/// copies memory from b into a
fn import_slice(target: &mut [f32], data:&[f32]) {
    target[..data.len()].copy_from_slice(data);
    target[data.len()..].fill(0f32);
}

fn test_reconstruct() {
    let (rows, cols, stride) = (8, 8, 8);
    let (s_x, s_y, s_z, s_t) = (rows, cols, cols, rows);
    debug_assert!(rows >= cols);
    let s_x = rows;
    let s_y = stride;
    let s_z = stride;
    let mut l_yt = generate_random_vector(rows * cols);
    // let mut t = generate_random_vector(cols * cols);
    let mut tri = create_identity_vector(cols, cols);

    let mut w = vec![0f32; cols];
    // these are x''s
    let mut x_argument = create_identity_vector(cols, cols);
    let mut o_buffer = x_argument.clone();
    let mut t_buffer = vec![0f32; rows * cols];
    import_slice(&mut t_buffer, &o_buffer);
    let mut s_buffer = vec![0f32; cols * cols];

    let input = l_yt.clone();

    wy_decomposition(&mut l_yt, &mut tri, &mut w, rows, cols, stride);

    let l_yt_matrix = NdArray {
        dims: vec![rows, cols],
        data: l_yt.clone(),
    };
    let tri_matrix = NdArray {
        dims: vec![rows, rows],
        data: tri.clone(),
    };
    println!("l_yt : {l_yt_matrix:?}");
    println!("tri : {tri_matrix:?}");

    // the compact WY representation: `A = L * (I - Y T Y')`.
    // A = LX - YTY'X;
    
    // y'x
    tensor_ut_contraction(
        &l_yt[1..],
        &o_buffer[stride..],
        &mut t_buffer,
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    let current = NdArray {
        dims: vec![rows, cols],
        data: t_buffer.clone(),
    };
    println!("check Y' created {current:?}");
    // t * [y'x];
    tensor_lt_contraction(
        &tri,
        &t_buffer,
        &mut s_buffer,
        1,
        0,
        rows,
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    let current = NdArray {
        dims: vec![rows, cols],
        data: s_buffer.clone(),
    };
    println!("check TY' created {current:?}");
    // VALIDATED AS CORRECT
    import_slice(&mut t_buffer, &s_buffer);
    // works and validated
    // [y']' * [ty'x ]
    tensor_tlt_contraction(
        &l_yt[1..],
        &s_buffer[..],
        &mut t_buffer[stride..],
        cols - cols.min(rows) + 1,
        0,
        rows.saturating_sub(1),
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    // THIS IS WHAT FAILS IE THE RHS TERM
    let mut q_argument = create_identity_vector(rows, cols);
    for k in 0..t_buffer.len() {
        q_argument[k] -= t_buffer[k];
    }
    let right_term = q_argument.clone();
    let right_term_matrix = NdArray {
        dims: vec![rows, cols],
        data: right_term.clone(),
    };
    println!("right_term {right_term_matrix:?}");
    // let mut t = create_identity_vector(cols, cols);
    t_buffer.fill(0f32);
    tensor_lt_contraction(
        &l_yt,
        &q_argument,
        &mut t_buffer,
        1,
        0,
        rows,
        cols,
        cols,
        stride,
        stride,
        stride,
    );
    let result = t_buffer.clone();
    let result_matrix = NdArray {
        dims: vec![rows, cols],
        data: result.clone(),
    };
    let input = NdArray {
        dims: vec![rows, cols],
        data: input.clone(),
    };
    println!("input : {input:?}");
    println!("reconstruct : {result_matrix:?}");
    // let reference = AutumnDecomp::new(input);
    // println!("reference LQ {:?}", reference.h);
    // println!("reference LQ {:?}", reference.t);
}

fn validate_upper_upper_fma() {
    let (rows, cols, stride) = (4, 7, 7);
    let mut d = generate_random_vector(rows * cols);
    let mut t = generate_random_vector(cols * cols);
    let mut w = vec![0f32; cols];

    let mut o_buffer = generate_random_vector(cols * cols);
    let mut t_buffer = o_buffer.clone();
    let mut t_clean = vec![0f32; cols * cols];
    // for testing
    let mut s_buffer = o_buffer.clone();

    let input = NdArray {
        dims: vec![cols, cols],
        data: o_buffer.clone(),
    };
    println!("input {input:?}");

    tensor_ut_contraction(
        &d[1..],
        &o_buffer[stride..],
        &mut t_clean[..],
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    tensor_ut_contraction(
        &d[1..],
        &o_buffer[stride..],
        &mut t_buffer[..],
        0,
        0,
        rows,
        cols.saturating_sub(1),
        cols,
        stride,
        stride,
        stride,
    );
    for k in 0..t_clean.len() {
        t_clean[k] += s_buffer[k];
    }
    println!("t_clean {t_clean:?}");
    println!("t_buffer {t_buffer:?}");

    for i in 0..rows {
        for j in 0..i.min(cols) {
            d[i * stride + j] = 0f32;
        }
        d[i * stride + i] = 1f32;
    }
    let basis_matrix = NdArray {
        dims: vec![rows, cols],
        data: d,
    };
    println!("basis_matrix {basis_matrix:?}");
    let s_vector = NdArray {
        dims: vec![cols, cols],
        data: s_buffer,
    };
    let reconst = NdArray {
        dims: vec![rows, cols],
        data: t_clean,
    };
    let reference = matrix_mult(&basis_matrix, &s_vector);
    println!("reconst {reconst:?}");
    println!("reference {reference:?}");
}

fn validate_transpose_upper_upper_fma() {
    // let (rows, cols, stride) = (4, 7, 7);
    let (rows, cols, stride) = (3, 3, 3);
    let mut d = generate_random_vector(cols * rows);
    let d_matrix = NdArray {
        dims: vec![cols, rows],
        data: d.clone(),
    };
    println!("raw x_matrix {d_matrix:?}");
    let mut w = vec![0f32; cols];

    // let mut o_buffer = vec![1f32; cols * cols];
    let mut o_buffer = generate_random_vector(cols * cols);
    let mut t_buffer = vec![0f32; rows * cols];
    import_slice(&mut t_buffer, &o_buffer);
    let mut t_clean = vec![0f32; rows * cols];
    // for testing
    let mut s_buffer = o_buffer.clone();

    let input = NdArray {
        dims: vec![cols, cols],
        data: o_buffer.clone(),
    };
    println!("input {input:?}");
    // tensor_tlt_contraction(
    //     &d[1..],
    //     &o_buffer[..],
    //     &mut t_clean[stride..],
    //     cols - cols.min(rows) + 1,
    //     0,
    //     rows.saturating_sub(1),
    //     cols,
    //     cols,
    //     rows,
    //     stride,
    //     stride,
    // );
    tensor_tlt_contraction(
        &d[1..],
        &o_buffer[..],
        &mut t_buffer[stride..],
        cols - cols.min(rows) + 1,
        0,
        rows.saturating_sub(1),
        cols,
        cols,
        rows,
        stride,
        stride,
    );
    // for k in 0..s_buffer.len() {
    //     t_clean[k] += s_buffer[k];
    // }
    // println!("t_clean {t_clean:?}");
    println!("t_buffer {t_buffer:?}");


    // let t_clean_mat = NdArray {
    //     dims: vec![rows, cols],
    //     data: t_clean.clone(),
    // };
    // println!("t_clean_mat {t_clean_mat:?}");
    // println!("----------------------");

    let mut basis_matrix = NdArray {
        dims: vec![cols, rows],
        data: d,
    };
    basis_matrix = basis_matrix.transpose();
    filter_lower_trapezoid(&mut basis_matrix);
    set_diagonal_value(&mut basis_matrix, 1f32);
    println!("basis_matrix {basis_matrix:?}");
    println!("----------------------");
    let s_vector = NdArray {
        dims: vec![cols, cols],
        data: s_buffer,
    };
    let reconst = NdArray {
        dims: vec![rows, cols],
        data: t_buffer,
    };
    let reference = matrix_mult(&basis_matrix, &s_vector);
    // println!("t_clean_mat {t_clean_mat:?}");
    println!("----------------------");
    println!("reconst {reconst:?}");
    println!("reference {reference:?}");
}

pub fn filter_lower_trapezoid(a: &mut NdArray) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let t = cols.min(rows);
    let s = rows.saturating_sub(cols);
    // don't remove from last row
    for i in 1..t {
        for j in 0..i {
            d[(rows - i - s) * cols - j - 1] = 0f32;
        }
    }
}
pub fn set_diagonal_value(a: &mut NdArray, c: f32) {
    let (rows, cols) = (a.dims[0], a.dims[1]);
    let d = &mut a.data;
    let mn = cols.min(rows);
    let dx = cols.saturating_sub(rows);
    for k in 0..mn {
        d[k * cols + dx + k] = c;
    }
}

fn debug_set_diagonal(rows:usize, cols:usize) {
    let mut d = generate_random_vector(rows * cols);
    let mut matrix = NdArray {
        dims: vec![rows, cols],
        data: d
    };
    filter_lower_trapezoid(&mut matrix);
    set_diagonal_value(&mut matrix, 1000f32);
    println!("set diagonal {matrix:?}");
}

fn test_debug_set_diagonal() {
    let vals = vec![(1,2), (2,1), (4,4), (3,6), (6,3)];
    for (r,c) in vals {
        debug_set_diagonal(r, c);
        println!("----------------------");
    }
}


// fn main() {
//     test_interview_intuition();
//     // test_reconstruct();
//     // test_reconstruct_transpose();
//     // validate_upper_upper_fma();
//     // validate_transpose_upper_upper_fma();
// }


 
 
 
fn generate_pattern_vector(n: usize, seed_offset: usize) -> Vec<f32> {
    // Avoid all-ones / all-same-constant data: LLVM can const-fold or the
    // kernel can hit unrepresentative cache behavior if every value is identical.
    // Cheap non-constant fill, no RNG throughput cost.
    (0..n)
        .map(|i| (((i + seed_offset) % 997) as f32) * 0.001 - 0.5)
        .collect()
}
fn bench_streamed_projection(mx: usize, mn: usize, chunk_rows: usize) {
    // Build the small projection matrix once (stand-in for Q / Y,T applied form).
    let proj_data = generate_pattern_vector(mn * mn, 0);
    let proj = NdArray {
        dims: vec![mn, mn],
        data: proj_data,
    };
 
    let mut out_chunk = vec![0f32; chunk_rows * mn];
    let num_full_chunks = mx / chunk_rows;
    let remainder = mx % chunk_rows;
 
    println!(
        "Streaming {} rows x {} cols in chunks of {} ({} full chunks, remainder {})",
        mx, mn, chunk_rows, num_full_chunks, remainder
    );
 
    let start = Instant::now();
 
    for c in 0..num_full_chunks {
        let y_data = generate_pattern_vector(chunk_rows * mn, c * 31 + 7);
        let y_chunk = NdArray {
            dims: vec![chunk_rows, mn],
            data: y_data,
        };
 
        for v in out_chunk.iter_mut() {
            *v = 0.0;
        }
        tensor_kernel(&y_chunk, &proj, &mut out_chunk);
 
        // touch the output so it isn't optimized away; in a real pipeline
        // this is where you'd write the chunk out or accumulate it
        std::hint::black_box(&out_chunk);
 
        if c % 10 == 0 {
            println!(
                "  chunk {}/{} done, elapsed {:.2?}",
                c + 1,
                num_full_chunks,
                start.elapsed()
            );
        }
    }
 
    if remainder > 0 {
        let y_data = generate_pattern_vector(remainder * mn, num_full_chunks * 31 + 7);
        let y_chunk = NdArray {
            dims: vec![remainder, mn],
            data: y_data,
        };
        let mut out_rem = vec![0f32; remainder * mn];
        tensor_kernel(&y_chunk, &proj, &mut out_rem);
        std::hint::black_box(&out_rem);
    }
 
    let elapsed = start.elapsed();
    println!("Total elapsed: {:.2?}", elapsed);
 
    let total_flops = 2.0 * (mx as f64) * (mn as f64) * (mn as f64);
    let gflops = total_flops / elapsed.as_secs_f64() / 1e9;
    println!(
        "Approx {:.3} GFLOPs total, {:.2} GFLOP/s effective",
        total_flops / 1e9,
        gflops
    );
}
 
fn main() {
    // Full interview-scale test: mx=5,000,000, mn=1,000
    // Start smaller first to sanity check before committing to the full run —
    // e.g. bench_streamed_projection(500_000, 1_000, 50_000);
    let mx = 5_000_000;
    let mn = 1_000;
    let chunk_rows = 20_000; // ~20_000*1000*4 bytes = ~80MB per chunk buffer
 
    bench_streamed_projection(mx, mn, chunk_rows);
}
 

