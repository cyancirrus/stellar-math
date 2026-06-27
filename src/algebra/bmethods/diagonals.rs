///  diagonal
///  - accumulates the multiplication into the target matrix
///  - t += x * y
///  - zero out t if u don't wish for accumulation
///
///   stride is always how the data is stored not a matrix dimension
#[inline(always)]
pub fn diagonal_lt(m: usize, p: usize, _n: usize) -> (usize, usize) {
    (p - p.min(m) + 1, 0)
}
#[inline(always)]
pub fn diagonal_ut(m: usize, p: usize, _n: usize) -> (usize, usize) {
    (0, m.saturating_sub(p))
}
#[inline(always)]
pub fn diagonal_rlt(_m: usize, p: usize, n: usize) -> (usize, usize) {
    (n - n.min(p), 0)
}
#[inline(always)]
pub fn diagonal_rut(_m: usize, p: usize, n: usize) -> (usize, usize) {
    (p - p.min(n) + 1, 0)
}
#[inline(always)]
pub fn diagonal_tlt(m: usize, p: usize, _n: usize) -> (usize, usize) {
    (p - p.min(m) + 1, 0)
}
#[inline(always)]
pub fn diagonal_tut(m: usize, p: usize, _n: usize) -> (usize, usize) {
    (0, m.saturating_sub(p))
}
