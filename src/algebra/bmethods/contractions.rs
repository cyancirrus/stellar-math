#![allow(unused)]
use crate::arch::SIMD_WIDTH;
use crate::structure::ndarray::NdArray;
use rayon::prelude::*;
use rayon::slice::ParallelSlice;
use std::cell::RefCell;
#[rustfmt::skip]
use crate::kernel::matkerns::{
    kernel_lt_mult,
    kernel_mult,
    kernel_rlt_mult,
    kernel_rut_mult,
    kernel_tlt_mult,
    kernel_tmult,
    kernel_tut_mult,
    kernel_ut_mult,
};

pub fn tensor_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..n).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
                kernel_mult(
                    x_d.get_unchecked(xoffset..),
                    y_d.get_unchecked(j..),
                    t_d.get_unchecked_mut(toffset + j..),
                    ii_end,
                    p,
                    jj_end,
                    s_x,
                    s_y,
                    s_t,
                );
            }
            toffset += dt;
            xoffset += dx;
        }
    }
}
pub fn tensor_tcontraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            for j in (0..n).step_by(SIMD_WIDTH) {
                let jj_end = SIMD_WIDTH.min(n - j);
                kernel_tmult(
                    x_d.get_unchecked(xoffset..),
                    y_d.get_unchecked(j..),
                    t_d.get_unchecked_mut(toffset + j..),
                    ii_end,
                    p,
                    jj_end,
                    s_x,
                    s_y,
                    s_t,
                );
            }
            toffset += dt;
            xoffset += dx;
        }
    }
}
pub fn tensor_lt_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    mut d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            if d_add + ii_end > d_sub {
                for j in (0..n).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(n - j);
                    kernel_lt_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                }
            }
            toffset += dt;
            xoffset += dx;
            d_add += SIMD_WIDTH;
        }
    }
}
pub fn tensor_ut_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    mut d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            if d_sub + p > d_add {
                for j in (0..n).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(n - j);
                    kernel_ut_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                }
            }
            toffset += dt;
            xoffset += dx;
            d_add += SIMD_WIDTH;
        }
    }
}
pub fn tensor_rlt_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    d_add: usize,
    mut d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for j in (0..n).step_by(SIMD_WIDTH) {
            let mut xoffset = 0;
            let mut toffset = 0;
            let jj_end = SIMD_WIDTH.min(n - j);
            // indexes the first zero
            if d_add + p > d_sub {
                for i in (0..m).step_by(SIMD_WIDTH) {
                    let ii_end = SIMD_WIDTH.min(m - i);
                    kernel_rlt_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    );
                    toffset += dt;
                    xoffset += dx;
                }
            }
            d_sub += SIMD_WIDTH;
        }
    }
}
pub fn tensor_rut_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    mut d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let dx = SIMD_WIDTH * s_x;
        let dt = SIMD_WIDTH * s_t;
        for j in (0..n).step_by(SIMD_WIDTH) {
            let mut xoffset = 0;
            let mut toffset = 0;
            let jj_end = SIMD_WIDTH.min(n - j);
            // indexes the first zero
            // if d_add + SIMD_WIDTH  > d_sub  {
            if d_add + jj_end > d_sub {
                for i in (0..m).step_by(SIMD_WIDTH) {
                    let ii_end = SIMD_WIDTH.min(m - i);
                    kernel_rut_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    );
                    toffset += dt;
                    xoffset += dx;
                }
            }
            d_add += SIMD_WIDTH;
        }
    }
}
pub fn tensor_tlt_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    mut d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            if d_add + ii_end > d_sub {
                for j in (0..n).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(n - j);
                    kernel_tlt_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                }
            }
            toffset += dt;
            xoffset += dx;
            d_add += SIMD_WIDTH;
        }
    }
}
pub fn tensor_tut_contraction(
    x_d: &[f32],
    y_d: &[f32],
    t_d: &mut [f32],
    mut d_add: usize,
    d_sub: usize,
    m: usize,
    p: usize,
    n: usize,
    s_x: usize,
    s_y: usize,
    s_t: usize,
) {
    unsafe {
        let mut xoffset = 0;
        let mut toffset = 0;
        let dx = SIMD_WIDTH;
        let dt = SIMD_WIDTH * s_t;
        for i in (0..m).step_by(SIMD_WIDTH) {
            let ii_end = SIMD_WIDTH.min(m - i);
            if d_sub + p > d_add {
                for j in (0..n).step_by(SIMD_WIDTH) {
                    let jj_end = SIMD_WIDTH.min(n - j);
                    kernel_tut_mult(
                        x_d.get_unchecked(xoffset..),
                        y_d.get_unchecked(j..),
                        t_d.get_unchecked_mut(toffset + j..),
                        d_add,
                        d_sub,
                        ii_end,
                        p,
                        jj_end,
                        s_x,
                        s_y,
                        s_t,
                    )
                }
            }
            toffset += dt;
            xoffset += dx;
            d_add += SIMD_WIDTH;
        }
    }
}
