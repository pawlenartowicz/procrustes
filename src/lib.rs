//! procrustes — orthogonal Procrustes, signed-permutation alignment, and
//! Generalised Procrustes Analysis (GPA).
//!
//! ## Convention
//!
//! All four alignment functions return a transform `T` such that
//! `a · T ≈ reference` minimizes the Frobenius norm under their respective
//! constraints. For [`orthogonal`] and [`rotation_only`], `T = R` is a
//! `K×K` orthogonal matrix (with [`rotation_only`] further restricted to
//! `det(R) = +1`). For [`signed_permutation`], `T = P · diag(signs)`
//! where `P` is the permutation encoded by `assigned`; equivalently,
//! column `k` of `a · T` equals `signs[k] · a[:, assigned[k]]`. For
//! [`sign_align`], `T = diag(signs)` (the degenerate identity-permutation
//! case). Matches `SciPy`'s `(A @ R) - B` minimization convention in
//! `scipy.linalg.orthogonal_procrustes`. For multi-matrix consensus
//! alignment, see [`generalized`].

#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub use faer::{Mat, MatRef};

mod gpa;
mod lap;
mod orthogonal;
mod signed_permutation;

pub use gpa::{generalized, GpaAlignment, GpaInit, GpaOptions, InnerAligner};
pub use orthogonal::{orthogonal, rotation_only, OrthogonalAlignment};
pub use signed_permutation::{sign_align, signed_permutation, SignAlignment, SignedPermutationAlignment};

/// `true` iff every entry of `x` is finite (neither NaN nor ±∞). Walks
/// column-major; short-circuits on the first non-finite value.
pub(crate) fn is_all_finite(x: MatRef<'_, f64>) -> bool {
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}

/// Error variants returned by the alignment functions ([`orthogonal`],
/// [`rotation_only`], [`signed_permutation`], [`sign_align`], [`generalized`]).
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
    /// At least one input dimension is zero.
    #[error("empty input: rows or columns is zero")]
    EmptyInput,
    /// Options validation failed; the message identifies the offending
    /// option (invalid weights, or `procrustes_form` with a zero-norm input).
    #[error("invalid options: {0}")]
    InvalidOptions(&'static str),
    /// `check_finite` was `true` and at least one input value was NaN or infinite.
    #[error("non-finite value in input")]
    NonFinite,
}
