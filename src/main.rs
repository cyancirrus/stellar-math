#![allow(unused)]
// TODO:
// then make the LX, async method
// do the 16 x 16 instruction ie 512 for the tower
// make the toml cfg to get cacheline size etc
// do a small test
// inspect the flamegraph to see if any hanging threads
// ie suspect like communication jam in l1-> l2
//
// value sanity start working on the LX async vision with the queue

// 1. Animate demo        ← most legible to employers
// 2. Blog redesign       ← makes everything else findable
// 3. Triangle kernel     ← 2hrs, unblocks LQ block
// 4. Trait refactor      ← important but least urgent

use std::ptr::copy_nonoverlapping;
use stellar::algebra::bmethods::pack;
use stellar::arch::SIMD_WIDTH;
use stellar::random::generation::generate_random_matrix;
use stellar_macros::{avx2_pack_simd_line_alligned, avx2_pack_simd_line_unalligned};
const MINIKERN_GATE: usize = SIMD_WIDTH * SIMD_WIDTH;
// const LC: usize = 48;
// const MC: usize = 48;
// const PC: usize = 32;
// const NC: usize = 96;
const MC: usize = 8;
const PC: usize = 8;
use std::arch::x86_64::{
    __m256, __m256i, _mm256_and_ps, _mm256_blendv_ps, _mm256_broadcast_ss, _mm256_castsi256_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_ps, _mm256_maskstore_ps,
    _mm256_storeu_ps, _mm256_setzero_ps
};
#[rustfmt::skip]
pub const MASK:[[i32;8];9] = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0,  0,  0],
    [-1, -1,  0,  0,  0,  0,  0,  0],
    [-1, -1, -1,  0,  0,  0,  0,  0],
    [-1, -1, -1, -1,  0,  0,  0,  0],
    [-1, -1, -1, -1, -1,  0,  0,  0],
    [-1, -1, -1, -1, -1, -1,  0,  0],
    [-1, -1, -1, -1, -1, -1, -1,  0],
    [-1, -1, -1, -1, -1, -1, -1, -1],
];
// #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
macro_rules! avx2_pack_x {
    ($bptr:expr, $dptr:expr, $re:expr, $se:expr, $s_b:expr, $s_d:expr) => {{
        let mut bptr = $bptr;
        let mut dptr = $dptr;
        let mut boffset: usize = 0;
        let mut doffset: usize = 0;
        let base_dptr = $dptr;
        if $se == PC {
            for _ in 0..$re {
                avx2_pack_simd_line_alligned!(bptr, dptr,);
                bptr = bptr.add(PC);
                dptr = dptr.add($s_d);
            }
        } else {
            for _ in 0..$re {
                avx2_pack_simd_line_unalligned!(bptr, dptr, $se, ZEROS);
                bptr = bptr.add(PC);
                dptr = dptr.add($s_d);
            }
        }
    }};
}
#[inline(always)]
#[cfg(not(any(feature = "avx2")))]
fn default_pack(bptr: *mut f32, dptr: *const f32, re: usize, se: usize, s_b: usize, s_d: usize) {
    unsafe {
        let mut doffset = 0;
        let mut boffset = 0;
        for _ in 0..re {
            copy_nonoverlapping(dptr.add(doffset), bptr.add(boffset), se);
            boffset += s_b;
            doffset += s_d;
        }
    }
}

macro_rules! pack_x {
    ($bptr:expr, $dptr:expr, $re:expr, $se:expr, $s_d:expr) => {{
        #[cfg(all(feature = "avx2", target_arch = "x86_64"))]
        avx2_pack_x!($bptr, $dptr, $re, $se, PC, $s_d);
        #[cfg(not(any(feature = "avx2")))]
        default_pack($bptr, $dptr, $re, $se, PC, $s_d);
    }};
}

use std::time::Instant;
// const MC: usize = 48;
// const PC: usize = 192;
// const NC: usize = 192;

// const PC: usize = 16;
// const NC: usize = 192;

#[inline(always)]
fn diff_min(x: usize, b: usize, t: usize) -> usize {
    if x - b < t { x - b } else { t }
}
fn test_performance() {
    let rows = 64;
    let cols = 40;
    // let rows = 64;
    // let cols = 256;
    let mut d = generate_random_matrix(rows, cols);
    let mut b_default = vec![0f32; MC * PC];
    let mut b_simd = vec![0f32; MC * PC];

    // let iters = 10_000;
    let iters = 100;

    // // // warmup
    // for _ in 0..100 {
    //     pack(&d.data, &mut b_default, MC, PC, PC, cols);
    // }
    println!("default");
    // default
    let start = Instant::now();
    for _ in 0..iters {
        for mc in (0..rows).step_by(MC) {
            let ma = diff_min(rows, mc, MC);
            for pc in (0..cols).step_by(PC) {
                let pa = diff_min(cols, pc, PC);
                b_default.fill(0f32);
                unsafe {
                    pack(&d.data[mc * cols + pc..], &mut b_default, ma, pa, PC, cols);
                }
                std::hint::black_box(&b_default);
            }
        }
    }
    let default_time = start.elapsed();
    println!("inline");
    // // simd
    let mut dptr = d.data.as_ptr();
    let start = Instant::now();
    for _ in 0..iters {
        for mc in (0..rows).step_by(MC) {
            let ma = diff_min(rows, mc, MC);
            for pc in (0..cols).step_by(PC) {
                let pa = diff_min(cols, pc, PC);
                unsafe {
                    pack_x!(
                        b_simd.as_mut_ptr(),
                        dptr.add(mc * cols + pc),
                        ma,
                        pa,
                        cols
                    );
                    std::hint::black_box(&b_simd);
                }
            }
        }
    }
    let simd_time = start.elapsed();

    // correctness
    assert_eq!(b_default, b_simd, "results don't match!");

    let total_calls = (iters * (rows / MC) * (cols / PC)) as u32;
    println!("default: {:?}", default_time );
    println!("simd:    {:?}", simd_time );
    println!(
        "speedup: {:.2}x",
        default_time.as_secs_f64() / simd_time.as_secs_f64()
    );
}

fn main() {
    test_performance();
}
