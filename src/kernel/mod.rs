#[cfg(all(feature = "avx512", target_arch = "x86_64"))]
pub mod avx512;

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
pub mod avx2;
pub mod avx2safe;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod neon;

pub mod default;
pub mod matkerns;
