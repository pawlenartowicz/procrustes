# procrustes

Orthogonal Procrustes (Schönemann SVD) and brute-force signed-permutation alignment for Rust, built on [`faer`](https://crates.io/crates/faer).

## Install

```toml
[dependencies]
procrustes = "0.1"
```

## When to use

- **`orthogonal`** — closed-form Schönemann SVD; cost dominated by the `K×K` SVD. Use whenever your alignment is a continuous rotation / reflection.
- **`signed_permutation`** — brute-force `O(K!·K)` enumeration with optimal sign per permutation. Practical up to about `K ≤ 10`; for `K ≳ 12` decompose by hand. Use when columns are abstractly indexed (PLS components, ICA sources, eigenmaps) and you need a discrete match rather than a rotation.
  - *TODO (planned, future minor version):* Hungarian-algorithm `O(K³)` fast path to lift the `K ≤ 10` ceiling. The cost matrix `C[i, j] = -|⟨a[:, i], reference[:, j]⟩|` is exactly a linear assignment problem after the closed-form sign reduction.

## Convention

Both functions return a transform `T` such that `a · T ≈ reference` minimizes the Frobenius norm under their respective constraints. For `orthogonal`, `T = R` is a `K×K` orthogonal matrix. For `signed_permutation`, `T = P · diag(signs)` where `P` is the permutation encoded by `assigned`; equivalently, column `k` of `a · T` equals `signs[k] · a[:, assigned[k]]`. Matches SciPy's `(A @ R) - B` minimization convention in `scipy.linalg.orthogonal_procrustes`.

## faer coupling

`MatRef<'_, f64>` and `Mat<f64>` appear in the public API. The crate re-exports them as `procrustes::{Mat, MatRef}`, so consumers do not need a separate `faer` dependency. Until faer reaches 1.0, **any faer major bump is a procrustes major bump** — the version pin is exact.

## Example

Recover a known rotation from a rotated copy of a reference matrix:

```rust
use procrustes::Mat;

let reference = Mat::<f64>::from_fn(4, 2, |i, j| match (i, j) {
    (0, 0) | (1, 1) => 1.0,
    (2, 0) | (3, 1) => 0.5,
    _ => 0.0,
});

// Apply a known 30° rotation in column space.
let theta = std::f64::consts::PI / 6.0;
let r0 = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
    (0, 0) | (1, 1) => theta.cos(),
    (0, 1) => -theta.sin(),
    (1, 0) => theta.sin(),
    _ => unreachable!(),
});
let a: Mat<f64> = &reference * &r0;

// Recover R = R0ᵀ.
let aln = procrustes::orthogonal(a.as_ref(), reference.as_ref(), true).unwrap();
let residual = aln.residual_frobenius(a.as_ref(), reference.as_ref());
assert!(residual < 1e-10);
```

For the discrete case, see `signed_permutation` in the [API docs](https://docs.rs/procrustes).

## License

Dual-licensed under `MIT OR Apache-2.0` at the user's option.

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
