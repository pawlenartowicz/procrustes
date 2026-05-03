# procrustes

[![crates.io](https://img.shields.io/crates/v/procrustes.svg)](https://crates.io/crates/procrustes)
[![docs.rs](https://img.shields.io/docsrs/procrustes/latest)](https://docs.rs/procrustes)
[![CI](https://github.com/pawlenartowicz/procrustes/actions/workflows/ci.yml/badge.svg)](https://github.com/pawlenartowicz/procrustes/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/procrustes.svg)](#license)

Orthogonal Procrustes (Schönemann SVD) and brute-force signed-permutation alignment for Rust, built on [`faer`](https://crates.io/crates/faer).

## Install

```toml
[dependencies]
procrustes = "0.1"
```

## When to use

- **`orthogonal`** — closed-form Schönemann SVD; `O(M·K² + K³)`. Use whenever your alignment is a continuous rotation / reflection.
- **`signed_permutation`** — exact discrete alignment of columns and per-column signs. Auto-routes by K: brute-force `O(K!·K)` enumeration for `K ≤ 8` (small-K bit-parity preserved), Jonker-Volgenant linear assignment `O(K³)` for `K ≥ 9` on the cost matrix `C[i, j] = -|⟨a[:, i], reference[:, j]⟩|`. Both paths return the global optimum. Use when columns are abstractly indexed (PLS components, ICA sources, eigenmaps) and you need a discrete match rather than a rotation.
- **`rotation_only`** — orthogonal Procrustes restricted to proper rotations (`det(R) = +1`, `R ∈ SO(K)`). Same SVD path as `orthogonal`; flips the last column of `U` if the SVD-derived rotation is a reflection. Use when reflection is physically meaningless (chemistry, physics, rigid-body alignment) or sign convention must be preserved across independent calls.
- **`sign_align`** — sign-only alignment, `O(M·K)` closed form. Per-column choice `s[k] = sign(⟨a[:, k], reference[:, k]⟩)`. Use when columns are already in the same canonical order as `reference` and only per-column sign is arbitrary — the typical PLS bootstrap pattern. For a general column-and-sign search, use `signed_permutation`.

## Convention

Both functions return a transform `T` such that `a · T ≈ reference` minimizes the Frobenius norm under their respective constraints. For `orthogonal`, `T = R` is a `K×K` orthogonal matrix. For `signed_permutation`, `T = P · diag(signs)` where `P` is the permutation encoded by `assigned`; equivalently, column `k` of `a · T` equals `signs[k] · a[:, assigned[k]]`. Matches SciPy's `(A @ R) - B` minimization convention in `scipy.linalg.orthogonal_procrustes`.

## faer coupling

`MatRef<'_, f64>` and `Mat<f64>` appear in the public API. The crate re-exports them as `procrustes::{Mat, MatRef}`, so consumers do not need a separate `faer` dependency. The pin is `faer = "0.24"` — caret-equivalent within the minor series — so patch bumps unify with downstream `^0.24` users automatically. Until faer reaches 1.0, **any faer minor bump (= breaking, pre-1.0) is a procrustes major bump**: a Dependabot watcher proposes the upgrade in a PR, and the bit-parity test in `tests/bit_parity.rs` is the tripwire that flags faer-side numerical drift even when the API still compiles.

## Third-party code

The Jonker-Volgenant LAP solver in `src/lap.rs` is a stripped-down port of
[Antti/lapjv-rust](https://github.com/Antti/lapjv-rust) v0.3.0 (MIT-licensed
by Andrii Dmytrenko). See `LICENSE-THIRDPARTY` for the full notice.

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
