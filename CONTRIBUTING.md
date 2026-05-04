# Contributing to `procrustes`

Thanks for your interest. This crate is single-maintainer and intentionally
small in scope; PRs are welcome but please open an issue first for anything
beyond a typo or a one-line bug fix.

## Running the test suite

```bash
cargo test --all-targets --locked
cargo test --doc --locked
```

Both must pass before you open a PR. CI runs both on stable, beta
(informational), and MSRV (`1.85`).

## Lint policy

Clippy and `missing_docs` are CI-enforced — PRs that introduce warnings will
fail. Run locally:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings -D rustdoc::broken_intra_doc_links -D rustdoc::private_intra_doc_links" \
    cargo doc --no-deps --locked
```

Note: `clippy::pedantic` is set to `warn` (not `deny`); pedantic warnings
are advisory locally and not gated by CI. `clippy::all` is denied.

## MSRV

Minimum Supported Rust Version is `1.85`. Bumping it requires:
- a `CHANGELOG.md` entry under the next version,
- a minor-version bump (`0.1.x → 0.2.0`),
- updating `rust-version` in `Cargo.toml` and the CI matrix in
  `.github/workflows/ci.yml`.

## faer policy

`faer` is caret-pinned (`"0.24"`, i.e. `>=0.24.0, <0.25.0`). Any faer
minor bump (`0.24 → 0.25`) is a procrustes major bump — see the README
"faer coupling" section. Do not loosen the bound in a PR without prior
discussion.

## Numerical tolerance convention

- Closed-form properties (e.g., orthogonality of `R`, `det(R) = ±1`):
  `1e-12` per element.
- SciPy parity: `1e-10` per element.

## Regenerating SciPy reference values

Each fixture in `tests/scipy_parity.rs` carries a Python regeneration recipe
in a `/* … */` comment immediately above the test function. To regenerate:
copy the recipe into a `python3` shell with `scipy ≥ 1.10` and `numpy ≥ 1.20`,
run, and paste the printed `R` and `scale` values back into the test
(reformatting to Rust array literals with explicit underscores per
`rustfmt`).

## What's intentionally out of scope

- `no_std` (faer is `std`-only).
- `f32` (public API is `f64`-only; revisit at `0.2.0` if requested).
- `serde` derivation on result types (file an issue if you need it).
- `ndarray` / `nalgebra` adapter feature flags (the only Rust competitor
  `scirs2-spatial` is ndarray-based; many users would benefit from a thin
  adapter). Convert at the call site for now —
  `Array2<f64>::as_slice()` + `MatRef::from_column_major_slice(...)` is
  one line. File an issue with a use case if you need first-class
  adapters; will be considered for `0.2.0`.
