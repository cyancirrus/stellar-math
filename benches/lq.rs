mod b_lq;
use b_lq::{decomposition, left_apply_q, left_apply_t, right_apply_q, right_apply_t};
mod sizes;
use criterion::{criterion_group, criterion_main};

criterion_group!(
    benches_apply,
    decomposition::bench_decomposition,
    left_apply_q::bench_apply_left_q,
    left_apply_q:: bench_apply_left_qt,
    right_apply_q::bench_apply_right_q,
    right_apply_q::bench_apply_right_qt,
    left_apply_t::bench_apply_left_t,
    right_apply_t::bench_apply_right_t,
);
criterion_main!(benches_apply);
