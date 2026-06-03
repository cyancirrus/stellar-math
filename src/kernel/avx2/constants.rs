// negative 1 is twos complement so all bits active
use std::arch::x86_64::{
    __m256, __m256i, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_maskload_ps, _mm256_maskstore_ps,
};
pub const SIMD_WIDTH:usize = 8;
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
#[rustfmt::skip]
pub const UMASK: [[i32; 8]; 9] = [
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [ 0, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  0, -1, -1, -1, -1, -1, -1],
    [ 0,  0,  0, -1, -1, -1, -1, -1],
    [ 0,  0,  0,  0, -1, -1, -1, -1],
    [ 0,  0,  0,  0,  0, -1, -1, -1],
    [ 0,  0,  0,  0,  0,  0, -1, -1],
    [ 0,  0,  0,  0,  0,  0,  0, -1],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
];
pub static ZEROS: [f32; 8] = [0f32; 8];
#[inline(always)]
pub unsafe fn mask_load(mask: __m256i, ptr: *const f32) -> __m256 {
    unsafe {
        _mm256_maskload_ps(ptr, mask)
        // _mm256_and_ps(_mm256_loadu_ps(ptr), _mm256_castsi256_ps(mask))
    }
}
#[inline(always)]
pub unsafe fn mask_load_ctrl(ctrl: i32, mask: __m256i, ptr: *const f32) -> __m256 {
    unsafe {
        let safe_ptr = if ctrl != 0 { ptr } else { ZEROS.as_ptr() };
        mask_load(mask, safe_ptr)
    }
}
#[inline(always)]
pub unsafe fn mask_store(mask: __m256i, tgt: *mut f32, data: __m256) {
    unsafe {
        _mm256_maskstore_ps(tgt, mask, data);
        // let out = _mm256_blendv_ps(_mm256_loadu_ps(tgt), data, _mm256_castsi256_ps(mask));
        // _mm256_storeu_ps(tgt, out);
    }
}
#[inline(always)]
pub unsafe fn mask_store_ctrl(ctrl: i32, mask: __m256i, tgt: *mut f32, data: __m256) {
    unsafe {
        if ctrl != 0 {
            mask_store(mask, tgt, data);
        }
    }
}
#[inline(always)]
pub fn cfma_accum(ctrl: i32, acc: __m256, sclr: *const f32, bout: __m256) -> __m256 {
    if ctrl != 0 {
        fma_accum(acc, sclr, bout)
    } else {
        acc
    }
}
#[inline(always)]
pub fn fma_accum(acc: __m256, sclr: *const f32, bout: __m256) -> __m256 {
    unsafe { _mm256_fmadd_ps(_mm256_broadcast_ss(&*sclr), bout, acc) }
}
