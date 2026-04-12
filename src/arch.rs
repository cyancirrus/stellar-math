#[cfg(feature = "avx512")]
pub const SIMD_WIDTH: usize = 16;
#[cfg(feature = "avx2")]
pub const SIMD_WIDTH: usize = 8;
#[cfg(not(any(feature = "avx2", feature = "avx512")))]
pub const SIMD_WIDTH: usize = 2;
