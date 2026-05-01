// negative 1 is twos complement so all bits active
use std::arch::x86_64::{
    __m256, __m256i, _mm256_and_ps, _mm256_blendv_ps, _mm256_broadcast_ss, _mm256_castsi256_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_maskload_ps, _mm256_maskstore_ps,
    _mm256_storeu_ps,
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

pub static ZEROS: [f32; 8] = [0f32; 8];
#[inline(always)]
pub unsafe fn mask_load(ptr: *const f32, mask: __m256i) -> __m256 {
    _mm256_maskload_ps(ptr, mask)
    // _mm256_and_ps(_mm256_loadu_ps(ptr), _mm256_castsi256_ps(mask))
}
#[inline(always)]
pub unsafe fn mask_load_ctrl(ptr: *const f32, mask: __m256i, ctrl: i32) -> __m256 {
    unsafe {
        let safe_ptr = if ctrl != 0 { ptr } else { ZEROS.as_ptr() };
        mask_load(safe_ptr, mask)
    }
}
#[inline(always)]
pub unsafe fn mask_store(tgt: *mut f32, mask: __m256i, data: __m256) {
    unsafe {
        _mm256_maskstore_ps(tgt, mask, data);
        // let out = _mm256_blendv_ps(_mm256_loadu_ps(tgt), data, _mm256_castsi256_ps(mask));
        // _mm256_storeu_ps(tgt, out);
    }
}
#[inline(always)]
pub unsafe fn mask_store_ctrl(tgt: *mut f32, mask: __m256i, data: __m256, ctrl: i32) {
    if ctrl != 0 {
        mask_store(tgt, mask, data);
    }
}
