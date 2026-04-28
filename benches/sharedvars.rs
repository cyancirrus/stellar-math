pub const LQ_SIZES: [usize; 4] = [8, 16, 32, 64];
pub const S_MATRIX_ALIGNED: [(usize, usize, usize); 4] =
    [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)];
pub const S_MATRIX_UNALIGNED: [(usize, usize, usize); 4] =
    [(4, 4, 4), (9, 9, 8), (31, 12, 48), (43, 53, 32)];
pub const M_MAT10_DIMS: [(usize, usize, usize); 4] = [
    (60, 128, 64),
    (120, 64, 256),
    (120, 256, 256),
    (250, 256, 256),
];
pub const M_MATRIX_DIMS: [(usize, usize, usize); 4] = [
    (64, 128, 64),
    (128, 64, 256),
    (128, 256, 256),
    (256, 256, 256),
];
pub const L_MATRIX_DIMS: [(usize, usize, usize); 4] = [
    (1024, 128, 1024),
    (64, 1024, 256),
    (512, 1024, 512),
    (1024, 1024, 1024),
];
pub const L_MAT10S_DIMS: [(usize, usize, usize); 4] = [
    (1020, 128, 1024),
    (60, 1024, 256),
    (510, 1024, 512),
    (1020, 1024, 1024),
];
