//! procrustes — orthogonal Procrustes & signed-permutation alignment.
//!
//! ## Convention
//!
//! Both [`orthogonal`] and [`signed_permutation`] return a transform `T` such
//! that `a · T ≈ reference` minimizes the Frobenius norm under their
//! respective constraints. For [`orthogonal`], `T = R` is a `K×K` orthogonal
//! matrix. For [`signed_permutation`], `T = P · diag(signs)` where `P` is
//! the permutation encoded by `assigned`; equivalently, column `k` of
//! `a · T` equals `signs[k] · a[:, assigned[k]]`. Matches `SciPy`'s
//! `(A @ R) - B` minimization convention in
//! `scipy.linalg.orthogonal_procrustes`.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub use faer::{Mat, MatRef};

mod lap;
mod orthogonal;
mod signed_permutation;

pub use orthogonal::{orthogonal, OrthogonalAlignment};
pub use signed_permutation::{signed_permutation, SignedPermutationAlignment};

/// Error variants for [`orthogonal`] and [`signed_permutation`].
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum ProcrustesError {
    /// `a` and `reference` differ in shape.
    #[error("dimension mismatch: a is {a_rows}×{a_cols}, reference is {ref_rows}×{ref_cols}")]
    DimensionMismatch {
        /// rows of `a`.
        a_rows: usize,
        /// columns of `a`.
        a_cols: usize,
        /// rows of `reference`.
        ref_rows: usize,
        /// columns of `reference`.
        ref_cols: usize,
    },
    /// `check_finite` was `true` and at least one input value was NaN or infinite.
    #[error("non-finite value in input")]
    NonFinite,
    /// At least one input dimension is zero.
    #[error("empty input: rows or columns is zero")]
    EmptyInput,
}
