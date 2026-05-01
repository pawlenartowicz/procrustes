# procrustes

Orthogonal Procrustes (Schönemann SVD) and brute-force signed-permutation alignment for Rust, built on [`faer`](https://crates.io/crates/faer).

## Convention

Both functions return a transform `T` such that `a · T ≈ reference` minimizes the Frobenius norm under their respective constraints. For `orthogonal`, `T = R` is a `K×K` orthogonal matrix. For `signed_permutation`, `T = P · diag(signs)` where `P` is the permutation encoded by `assigned`; equivalently, column `k` of `a · T` equals `signs[k] · a[:, assigned[k]]`. Matches SciPy's `(A @ R) - B` minimization convention in `scipy.linalg.orthogonal_procrustes`.

## faer coupling

`MatRef<'_, f64>` and `Mat<f64>` appear in the public API. The crate re-exports them as `procrustes::{Mat, MatRef}`, so consumers do not need a separate `faer` dependency. Until faer reaches 1.0, **any faer major bump is a procrustes major bump** — the version pin is exact.

## Example

```rust
use procrustes::Mat;

let a = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
let reference = a.clone();

let alignment = procrustes::orthogonal(a.as_ref(), reference.as_ref(), true).unwrap();
assert!((alignment.scale - 2.0).abs() < 1e-10);

let perm = procrustes::signed_permutation(a.as_ref(), reference.as_ref(), true).unwrap();
assert_eq!(perm.assigned, vec![0, 1]);
assert_eq!(perm.signs, vec![1.0, 1.0]);
```

## License

Dual-licensed under `MIT OR Apache-2.0` at the user's option.

---
Paweł Lenartowicz · [Freestyler Scientists](https://freestylerscientists.pl) · [github.com/pawlenartowicz](https://github.com/pawlenartowicz/)
