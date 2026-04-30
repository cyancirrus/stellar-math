// negative 1 is twos complement so all bits active
use std::arch::x86_64::{
    __m256, __m256i, _MM_HINT_T0, _mm_prefetch, _mm256_add_ps, _mm256_and_ps, _mm256_broadcast_ss,
    _mm256_castsi256_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256, _mm256_maskload_ps,
    _mm256_maskstore_ps, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps,
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
pub unsafe fn gate_row(ptr: *const f32, ctrl: i32, mask: __m256i) -> __m256 {
    unsafe {
        let safe_ptr = if ctrl != 0 { ptr } else { ZEROS.as_ptr() };
        _mm256_maskload_ps(safe_ptr, mask)
    }
}
pub unsafe fn sgate_row(ptr: *mut f32, ctrl: i32, mask: __m256i, data: __m256) {
    unsafe {
        if ctrl != 0 {
            _mm256_maskstore_ps(ptr, mask, data);
        }
    }
}
macro_rules! fma_gated {
    ($acc:expr, $ptr:expr, $mask_bit:expr, $data:expr) => {
        if $mask_bit != 0 {
            $acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&*$ptr), $data, $acc);
            // $acc = _mm256_fmadd_ps(_mm256_set1_ps(*$ptr), $b, $acc);
        }
    };
}
macro_rules! fma_accum {
    ($acc:expr, $ptr:expr, $data:expr) => {
        $acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&*$ptr), $data, $acc);
    };
}
