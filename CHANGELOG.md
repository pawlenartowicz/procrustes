# Changelog

All notable changes to this crate will be documented in this file.

## [0.1.1] — 2026-05-04

- Added: `generalized(matrices, opts) -> Result<GpaAlignment, _>` — iterative
  consensus alignment of `N` matrices. Public surface: `generalized`,
  `GpaAlignment`, `GpaOptions`, `GpaInit`, `InnerAligner`. Inner aligner is
  configurable (`Orthogonal` / `SignedPermutation` / `RotationOnly`); init
  is `FirstMatrix` (default) or `Mean`; supports per-matrix scalar weights
  and optional Procrustes-form pre-scaling. Convergence on consensus drift,
  default `tol = 1e-10`, `max_iters = 100`.
- Added: `ProcrustesError::InvalidOptions(&'static str)` variant for options
  validation errors in `generalized`.
- Added: Jonker-Volgenant `O(K³)` fast path inside `signed_permutation` for
  `K ≥ 9` (brute force preserved for `K ≤ 8`). Public signature unchanged.
  Both paths return the global optimum; permutation at exact cost ties is
  implementation-defined.
- Added: `LICENSE-THIRDPARTY` documenting the MIT-licensed JV core ported
  from `Antti/lapjv-rust` v0.3.0.
- Out of scope (deferred to `0.2.0+` on user request): `ndarray` / `nalgebra`
  adapter feature flags. File an issue if needed.
- Added: `sign_align(a, reference, check_finite) -> Result<SignAlignment, _>`
  for sign-only alignment (degenerate `signed_permutation` with identity
  permutation). `O(M·K)`, intended for PLS bootstrap and similar workflows
  where column order is canonical. Result exposes `signs` and eager
  `residual_frobenius` (un-squared, matching `SignedPermutationAlignment`).
- Added: `rotation_only(a, reference, check_finite) -> Result<OrthogonalAlignment, _>` —
  orthogonal Procrustes restricted to `SO(K)` (proper rotations,
  `det(R) = +1`). Flips the last column of `U` from the SVD when the
  unconstrained rotation is a reflection. `orthogonal` is unchanged.
- Added: Criterion-based wall-time benches in `benches/alignment.rs` covering
  all four alignment entry points (`orthogonal`, `rotation_only`,
  `signed_permutation`, `sign_align`) over `K ∈ {2, 3, 4, 6, 10, 16}` at
  `M = 32`. The `signed_permutation` sweep crosses `BRUTE_FORCE_CUTOFF` so
  the brute-force and Jonker-Volgenant paths are both exercised. CI runs
  `cargo bench --no-run --locked` as a compile-check; actual measurements
  are local-only (runners are too noisy).
- `deny.toml` introduced (advisories / licenses / bans / sources). CI runs
  `cargo deny check` with `continue-on-error: true` for one release cycle;
  promote to required in `0.1.3` or `0.2.0`.

## [0.1.0] — 2026-05-02

Initial release. Public surface: `orthogonal`, `signed_permutation`, plus result and error types. Faer 0.24 in the public API.
