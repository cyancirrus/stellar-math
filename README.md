# StellarMath

StellarMath is a Rust numerical library exploring linear algebra, matrix decompositions, Fourier transforms, and related algorithms.  
The project was built from scratch to gain a deep understanding of the underlying mathematics, array layouts, and algorithmic design.

---

## Motivation

This library is designed as a hands-on learning tool and experimentation ground:  
- Implement numerical methods from scratch (SVD, Cholesky, LU, QR, FFT, etc.)  
- Explore matrix and tensor layouts (row-major, contiguous memory, strides)  
- Evaluate performance and potential optimizations for Rust  
- Provide examples and playground for community feedback  

---

## Library Scope & Boundaries

- Current decomposition algorithms (Cholesky, LU, QR, SVD, etc.) operate on **2D arrays only**.  
- Arrays are row-major and store data in contiguous memory for simplicity.  
- Future plans:  
  - Extend decomposition and solver methods to **N-dimensional arrays** using strides.  
  - Optionally integrate with [`ndarray`](https://docs.rs/ndarray/latest/ndarray/) for tensor operations.  
- The library is intentionally low-level, with minimal dependencies, to give fine-grained control over memory and computation.

---

## Installation

Add StellarMath to your `Cargo.toml`:

```toml
[dependencies]
stellar_math = { path = "path/to/stellar_math" }
````

---

## Examples

You can find runnable examples in the `examples/` directory:

```bash
cargo run --example svd_demo
cargo run --example fft_demo
cargo run --example cholesky_demo
```

These examples demonstrate how to use various decompositions, solvers, and transforms, and print results in a readable matrix format.

---

## Benchmarking

Create a benchmark:

```bash
cargo flamegraph --root --bench matrix_benchmark
```

View the flamegraph for benchmark performance:

```bash
open -a firefox flamegraph.svg
```

---

## Feedback

I’m especially interested in feedback on:

* Which parts of the library might be **most useful to the Rust numerical community**
* Opportunities to **integrate with well-known packages** like `ndarray`
* Suggestions for improving **readability, API ergonomics, or performance**
* Any additional features or extensions that would be valuable

All feedback is welcome — this library is meant to evolve with community input.

---

## License

MIT or Apache 2.0 (choose whichever fits your goals)
