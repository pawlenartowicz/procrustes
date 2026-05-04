//! Generalized Procrustes Analysis (GPA): iterative consensus alignment
//! of `N` matrices via repeated inner Procrustes calls.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

use crate::ProcrustesError;

/// Inner per-iteration aligner used by [`generalized`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum InnerAligner {
    /// [`crate::orthogonal`] — Schönemann SVD; `R ∈ O(K)`.
    /// Default; matches morphometric / shape-analysis convention.
    Orthogonal,
    /// [`crate::signed_permutation`] — exact signed-permutation alignment.
    /// Recommended for PLS bootstrap, where component order can vary
    /// across resamples.
    SignedPermutation,
    /// [`crate::rotation_only`] — `R ∈ SO(K)` (proper rotation, no
    /// reflection). Use when reflection is physically meaningless
    /// (chemistry, rigid-body alignment).
    RotationOnly,
}

/// Initial consensus seed for [`generalized`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum GpaInit {
    /// Use `matrices[0]` (post-`procrustes_form` rescaling, if enabled) as
    /// the initial consensus. Result depends on input ordering; converges
    /// in fewer iterations when inputs are already similar. Canonical for
    /// PLS bootstrap (fixed iteration order).
    FirstMatrix,
    /// Use the (weighted, if `weights` are set) arithmetic mean of the raw
    /// inputs as the initial consensus. Order-symmetric but rotation-
    /// sensitive on un-pre-aligned data: when inputs differ by independent
    /// uniform rotations, the mean collapses toward zero as `N` grows,
    /// giving a degenerate seed. Pair with `procrustes_form = true` or use
    /// [`GpaInit::FirstMatrix`] when this is a risk.
    Mean,
}

/// Options for [`generalized`]. Construct via `GpaOptions::default()` and
/// override fields with struct-update syntax.
#[derive(Debug, Clone)]
pub struct GpaOptions {
    /// Inner aligner. Default [`InnerAligner::Orthogonal`].
    pub inner: InnerAligner,
    /// Initial consensus seed. Default [`GpaInit::FirstMatrix`].
    pub init: GpaInit,
    /// Convergence tolerance on consensus drift `‖M_t − M_{t-1}‖_F`,
    /// absolute. Default `1e-10`.
    pub tol: f64,
    /// Maximum iterations before returning with `converged = false`.
    /// Default `100`. Setting `max_iters = 0` is a degenerate case: the
    /// loop never executes, so [`GpaAlignment::aligned`] is empty, and
    /// [`GpaAlignment::final_drift`] is `f64::NAN`.
    pub max_iters: usize,
    /// If `true`, pre-scale each input to `‖A_i‖_F = 1` before the
    /// alignment loop (shape-analysis convention). Default `false`.
    pub procrustes_form: bool,
    /// Optional per-matrix scalar weights for the consensus update
    /// `M = Σ w_i · Ã_i / Σ w_i`. `None` → uniform `1/N`. When `Some`,
    /// must have length `matrices.len()`, all entries finite and `≥ 0`,
    /// with at least one strictly positive entry.
    pub weights: Option<Vec<f64>>,
}

impl Default for GpaOptions {
    fn default() -> Self {
        Self {
            inner: InnerAligner::Orthogonal,
            init: GpaInit::FirstMatrix,
            tol: 1e-10,
            max_iters: 100,
            procrustes_form: false,
            weights: None,
        }
    }
}

/// Result of [`generalized`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct GpaAlignment {
    /// Final consensus matrix `M` (`M × K`).
    pub consensus: Mat<f64>,
    /// Per-input aligned matrix `Ã_i` against the final consensus
    /// (length `N`, each `M × K`). When `procrustes_form` was set, these
    /// are aligned in the unit-Frobenius space, **not** the original
    /// magnitudes. Empty when [`GpaOptions::max_iters`] is `0`.
    pub aligned: Vec<Mat<f64>>,
    /// Number of full iterations performed (one alignment pass + one
    /// consensus update = one iteration).
    pub n_iters: usize,
    /// Final consensus drift `‖M_t − M_{t-1}‖_F` at termination.
    pub final_drift: f64,
    /// `true` if `final_drift < opts.tol`; `false` if `max_iters` was hit
    /// before reaching tol.
    pub converged: bool,
}

/// Generalised Procrustes Analysis: iterative consensus alignment of `N`
/// matrices.
///
/// Given `matrices = [A_1, …, A_N]` of identical shape `M × K`, repeatedly
///
/// 1. align each `A_i` to the current consensus `M_t` via the chosen
///    [`InnerAligner`], producing `Ã_i`,
/// 2. recompute `M_{t+1}` as the (weighted) mean of `{Ã_i}`,
///
/// until `‖M_{t+1} − M_t‖_F < opts.tol` or `n_iters == opts.max_iters`.
///
/// Inputs are not finiteness-checked — sanitise upstream or wrap a single
/// `crate::orthogonal(_, _, true)` call to validate before invoking.
///
/// # Errors
/// - [`ProcrustesError::EmptyInput`] when `matrices` is empty or
///   `matrices[0]` has a zero dimension.
/// - [`ProcrustesError::DimensionMismatch`] when matrices differ in shape.
/// - [`ProcrustesError::InvalidOptions`] when weights are invalid (wrong
///   length, non-finite, negative, or zero-sum) or `procrustes_form = true`
///   with any input of F-norm `< f64::EPSILON`.
///
/// # Examples
/// ```
/// use procrustes::{generalized, GpaOptions, Mat};
/// let a = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
/// let mats = [a.as_ref(), a.as_ref(), a.as_ref()];
/// let aln = generalized(&mats, GpaOptions::default()).unwrap();
/// assert!(aln.converged);
/// // Identical inputs ⇒ consensus equals the input.
/// assert!((aln.consensus[(0, 0)] - 1.0).abs() < 1e-12);
/// ```
#[allow(clippy::many_single_char_names)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::too_many_lines)]
pub fn generalized(
    matrices: &[MatRef<'_, f64>],
    opts: GpaOptions,
) -> Result<GpaAlignment, ProcrustesError> {
    // --- Validation ---

    if matrices.is_empty() {
        return Err(ProcrustesError::EmptyInput);
    }

    let (ref_rows, ref_cols) = (matrices[0].nrows(), matrices[0].ncols());
    if ref_rows == 0 || ref_cols == 0 {
        return Err(ProcrustesError::EmptyInput);
    }

    for m in matrices.iter().skip(1) {
        if m.nrows() != ref_rows || m.ncols() != ref_cols {
            return Err(ProcrustesError::DimensionMismatch {
                a_rows: m.nrows(),
                a_cols: m.ncols(),
                ref_rows,
                ref_cols,
            });
        }
    }

    if let Some(w) = &opts.weights {
        if w.len() != matrices.len() {
            return Err(ProcrustesError::InvalidOptions(
                "weights length must equal matrices length",
            ));
        }
        for &wi in w {
            if !wi.is_finite() {
                return Err(ProcrustesError::InvalidOptions(
                    "weights contain non-finite value",
                ));
            }
            if wi < 0.0 {
                return Err(ProcrustesError::InvalidOptions(
                    "weights contain negative value",
                ));
            }
        }
        let sum: f64 = w.iter().sum();
        if sum <= 0.0 {
            return Err(ProcrustesError::InvalidOptions("weights sum to zero"));
        }
    }

    // Pre-scale to Procrustes form if requested, checking for zero-norm inputs.
    // `scaled_storage` and `scaled_refs` must outlive `inputs` because the
    // MatRefs in `scaled_refs` borrow from `scaled_storage`.
    let scaled_storage: Vec<Mat<f64>> = if opts.procrustes_form {
        let mut storage = Vec::with_capacity(matrices.len());
        for &m in matrices {
            let norm = frobenius(m);
            if norm < f64::EPSILON {
                return Err(ProcrustesError::InvalidOptions(
                    "procrustes_form requires non-zero inputs",
                ));
            }
            let inv = 1.0 / norm;
            storage.push(Mat::<f64>::from_fn(ref_rows, ref_cols, |i, j| {
                m[(i, j)] * inv
            }));
        }
        storage
    } else {
        Vec::new()
    };
    let scaled_refs: Vec<MatRef<'_, f64>> = scaled_storage.iter().map(Mat::as_ref).collect();
    // Borrow `matrices` directly when no rescaling is needed; only the rescaled
    // path requires the owned `scaled_refs` Vec.
    let inputs: &[MatRef<'_, f64>] = if opts.procrustes_form {
        &scaled_refs
    } else {
        matrices
    };

    // --- Initialise consensus ---
    let mut consensus = match opts.init {
        GpaInit::FirstMatrix => {
            let m = inputs[0];
            Mat::<f64>::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)])
        }
        GpaInit::Mean => weighted_mean(inputs, opts.weights.as_deref()),
    };

    // Scratch for the last aligned snapshot (returned on loop exhaustion).
    let mut last_aligned: Vec<Mat<f64>> = Vec::new();
    let mut last_drift = f64::NAN;

    // --- Iterative alignment ---
    for iter in 0..opts.max_iters {
        let aligned: Vec<Mat<f64>> = inputs
            .iter()
            .map(|&m| apply_inner(m, consensus.as_ref(), opts.inner))
            .collect();

        let aligned_refs: Vec<MatRef<'_, f64>> = aligned.iter().map(Mat::as_ref).collect();
        let new_consensus = weighted_mean(&aligned_refs, opts.weights.as_deref());
        let drift = frobenius_diff(new_consensus.as_ref(), consensus.as_ref());
        consensus = new_consensus;

        if drift < opts.tol {
            return Ok(GpaAlignment {
                consensus,
                aligned,
                n_iters: iter + 1,
                final_drift: drift,
                converged: true,
            });
        }

        last_aligned = aligned;
        last_drift = drift;
    }

    // Loop exhausted.
    Ok(GpaAlignment {
        consensus,
        aligned: last_aligned,
        n_iters: opts.max_iters,
        final_drift: last_drift,
        converged: false,
    })
}

/// Apply the chosen inner aligner: returns a fresh `Ã = matrix · T` where
/// `T` is the transform produced by [`InnerAligner`] against `reference`.
#[allow(clippy::many_single_char_names)]
fn apply_inner(
    matrix: MatRef<'_, f64>,
    reference: MatRef<'_, f64>,
    inner: InnerAligner,
) -> Mat<f64> {
    // Inner errors are unreachable here: shape and finiteness validation is
    // already done upstream in `generalized`. SVD failure NaN-fills (per the
    // existing inner-aligner contract); NaN propagates to the consensus and
    // `final_drift`, so we'll spin to `max_iters` and return `converged =
    // false` with a NaN consensus.
    let (m, k) = (matrix.nrows(), matrix.ncols());
    match inner {
        InnerAligner::Orthogonal => {
            let aln = crate::orthogonal(matrix, reference, false).expect("validated upstream");
            matmul_apply(matrix, aln.rotation.as_ref())
        }
        InnerAligner::RotationOnly => {
            let aln = crate::rotation_only(matrix, reference, false).expect("validated upstream");
            matmul_apply(matrix, aln.rotation.as_ref())
        }
        InnerAligner::SignedPermutation => {
            let aln =
                crate::signed_permutation(matrix, reference, false).expect("validated upstream");
            // Ã[:, k] = signs[k] · matrix[:, assigned[k]]. Done by hand
            // rather than via `matmul_apply` to avoid materialising a dense
            // K×K signed-permutation matrix; cost is O(M·K) vs O(M·K²).
            let mut out = Mat::<f64>::zeros(m, k);
            for kk in 0..k {
                let s = aln.signs[kk];
                let src = aln.assigned[kk];
                for ii in 0..m {
                    out[(ii, kk)] = s * matrix[(ii, src)];
                }
            }
            out
        }
    }
}

/// Compute `matrix · rot` into a fresh `M × K` buffer.
fn matmul_apply(matrix: MatRef<'_, f64>, rot: MatRef<'_, f64>) -> Mat<f64> {
    let (m, k) = (matrix.nrows(), matrix.ncols());
    let mut out = Mat::<f64>::zeros(m, k);
    matmul(out.as_mut(), Accum::Replace, matrix, rot, 1.0, Par::Seq);
    out
}

/// Compute `(Σ w_i · M_i) / Σ w_i` with a single allocation.
/// `weights = None` → uniform `1/N`.
///
/// **Precondition:** `matrices` is non-empty and all entries share shape.
/// Caller (only [`generalized`]) enforces this upstream; passing an empty
/// slice panics on `matrices[0]`.
#[allow(clippy::many_single_char_names)]
#[allow(clippy::cast_precision_loss)]
fn weighted_mean(matrices: &[MatRef<'_, f64>], weights: Option<&[f64]>) -> Mat<f64> {
    let n = matrices.len();
    let (rows, cols) = (matrices[0].nrows(), matrices[0].ncols());

    let mut out = Mat::<f64>::zeros(rows, cols);

    match weights {
        None => {
            let scale = 1.0 / (n as f64);
            for &m in matrices {
                for j in 0..cols {
                    for r in 0..rows {
                        out[(r, j)] += scale * m[(r, j)];
                    }
                }
            }
        }
        Some(w) => {
            let inv = 1.0 / w.iter().sum::<f64>();
            for (&wi, &m) in w.iter().zip(matrices.iter()) {
                if wi == 0.0 {
                    continue;
                }
                let scale = wi * inv;
                for j in 0..cols {
                    for r in 0..rows {
                        out[(r, j)] += scale * m[(r, j)];
                    }
                }
            }
        }
    }

    out
}

/// `√Σ (a_ij − b_ij)²` — Frobenius distance between two same-shape matrices.
fn frobenius_diff(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = a[(i, j)] - b[(i, j)];
            s += d * d;
        }
    }
    s.sqrt()
}

/// `√Σ x_ij²` — Frobenius norm of a matrix.
fn frobenius(x: MatRef<'_, f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            let v = x[(i, j)];
            s += v * v;
        }
    }
    s.sqrt()
}
