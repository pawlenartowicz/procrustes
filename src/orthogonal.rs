//! Continuous orthogonal Procrustes alignment via Schönemann SVD.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

use crate::{is_all_finite, ProcrustesError};

/// Solve `min_R ‖a · R − reference‖_F` over orthogonal `K×K` `R`.
///
/// Closed-form Schönemann SVD: with `M = aᵀ · reference = U Σ Vᵀ`,
/// the optimum is `R = U · Vᵀ`.
///
/// # Errors
/// - [`ProcrustesError::DimensionMismatch`] if `a` and `reference` differ
///   in rows or columns.
/// - [`ProcrustesError::EmptyInput`] if either dimension is zero.
/// - [`ProcrustesError::NonFinite`] if `check_finite` is `true` and any
///   input value is NaN or infinite.
///
/// # Examples
/// ```
/// use procrustes::Mat;
/// let a = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
/// let reference = a.clone();
/// let alignment = procrustes::orthogonal(a.as_ref(), reference.as_ref(), true).unwrap();
/// // a equals reference, so the rotation is the identity and scale = sum(svd(I₂)) = 2.
/// assert!((alignment.scale - 2.0).abs() < 1e-10);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn orthogonal(
    a: MatRef<'_, f64>,
    reference: MatRef<'_, f64>,
    check_finite: bool,
) -> Result<OrthogonalAlignment, ProcrustesError> {
    let (a_rows, a_cols) = (a.nrows(), a.ncols());
    let (ref_rows, ref_cols) = (reference.nrows(), reference.ncols());

    if a_rows != ref_rows || a_cols != ref_cols {
        return Err(ProcrustesError::DimensionMismatch {
            a_rows,
            a_cols,
            ref_rows,
            ref_cols,
        });
    }
    if a_rows == 0 || a_cols == 0 {
        return Err(ProcrustesError::EmptyInput);
    }
    if check_finite && (!is_all_finite(a) || !is_all_finite(reference)) {
        return Err(ProcrustesError::NonFinite);
    }

    let k = a_cols;

    // M_buf = aᵀ · reference  (K × K)
    let mut m_buf = Mat::<f64>::zeros(k, k);
    matmul(
        m_buf.as_mut(),
        Accum::Replace,
        a.transpose(),
        reference,
        1.0,
        Par::Seq,
    );

    // SVD: M_buf = U · diag(s) · Vᵀ. With finite inputs validated upstream the
    // SVD always converges; the only path to failure here is `check_finite =
    // false` plus NaN/inf inputs, where the spec contract is "result is
    // undefined — just don't panic". On SVD failure we therefore return a
    // NaN-filled rotation rather than panicking.
    let Ok(svd) = m_buf.as_ref().svd() else {
        return Ok(OrthogonalAlignment {
            rotation: Mat::<f64>::from_fn(k, k, |_, _| f64::NAN),
            scale: f64::NAN,
        });
    };
    let u = svd.U();
    let v = svd.V();

    // R = U · Vᵀ  (K × K orthogonal).
    let mut rotation = Mat::<f64>::zeros(k, k);
    matmul(
        rotation.as_mut(),
        Accum::Replace,
        u,
        v.transpose(),
        1.0,
        Par::Seq,
    );

    // scale = nuclear norm ‖M_buf‖_* = sum of singular values
    //       = trace(M_buf · Rᵀ) = sum_{i,j} M_buf[i,j] · R[i,j]   (O(K²)).
    let mut scale = 0.0;
    for i in 0..k {
        for j in 0..k {
            scale += m_buf[(i, j)] * rotation[(i, j)];
        }
    }

    Ok(OrthogonalAlignment { rotation, scale })
}

/// Orthogonal Procrustes restricted to proper rotations (`det(R) = +1`,
/// `R ∈ SO(K)`).
///
/// Identical to [`orthogonal`] except: if the SVD-derived rotation has
/// `det(R) = -1` (i.e. is a reflection), flip the sign of the last column
/// of `U` before forming `R`. The returned rotation is then guaranteed
/// proper at the cost of a typically small increase in residual.
///
/// Use this when reflection is physically meaningless (chemistry, physics,
/// rigid-body alignment) or when sign convention must be preserved across
/// independent calls.
///
/// At `K = 1`, `SO(1) = {[[1.0]]}`, so the returned rotation is always the
/// identity regardless of input — even when a sign flip would minimize the
/// residual.
///
/// `scale` in the returned [`OrthogonalAlignment`] is recomputed against
/// the (possibly flipped) `R`, so [`OrthogonalAlignment::residual_frobenius`]
/// remains valid downstream.
///
/// # Errors
/// Same as [`orthogonal`]:
/// [`ProcrustesError::DimensionMismatch`] on shape mismatch,
/// [`ProcrustesError::EmptyInput`] if either dimension is zero,
/// [`ProcrustesError::NonFinite`] if `check_finite` is `true` and any input
/// value is NaN or infinite.
///
/// # Examples
/// ```
/// use procrustes::Mat;
/// // a == reference, so the rotation is the identity (det = +1).
/// let a = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
/// let aln = procrustes::rotation_only(a.as_ref(), a.as_ref(), true).unwrap();
/// // Identity is a proper rotation; residual is zero.
/// assert!(aln.residual_frobenius(a.as_ref(), a.as_ref()) < 1e-10);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn rotation_only(
    a: MatRef<'_, f64>,
    reference: MatRef<'_, f64>,
    check_finite: bool,
) -> Result<OrthogonalAlignment, ProcrustesError> {
    let (a_rows, a_cols) = (a.nrows(), a.ncols());
    let (ref_rows, ref_cols) = (reference.nrows(), reference.ncols());

    if a_rows != ref_rows || a_cols != ref_cols {
        return Err(ProcrustesError::DimensionMismatch {
            a_rows,
            a_cols,
            ref_rows,
            ref_cols,
        });
    }
    if a_rows == 0 || a_cols == 0 {
        return Err(ProcrustesError::EmptyInput);
    }
    if check_finite && (!is_all_finite(a) || !is_all_finite(reference)) {
        return Err(ProcrustesError::NonFinite);
    }

    let k = a_cols;

    // M = aᵀ · reference  (K × K)
    let mut m_buf = Mat::<f64>::zeros(k, k);
    matmul(
        m_buf.as_mut(),
        Accum::Replace,
        a.transpose(),
        reference,
        1.0,
        Par::Seq,
    );

    // SVD: M = U · diag(s) · Vᵀ. As in `orthogonal`, fall back to a NaN
    // result on SVD failure rather than panicking.
    let Ok(svd) = m_buf.as_ref().svd() else {
        return Ok(OrthogonalAlignment {
            rotation: Mat::<f64>::from_fn(k, k, |_, _| f64::NAN),
            scale: f64::NAN,
        });
    };
    let u = svd.U();
    let v = svd.V();

    // Initial R = U · Vᵀ  (orthogonal, may be reflection or rotation).
    let mut rotation = Mat::<f64>::zeros(k, k);
    matmul(
        rotation.as_mut(),
        Accum::Replace,
        u,
        v.transpose(),
        1.0,
        Par::Seq,
    );

    // Detect reflection via sign of det(R). For an orthogonal R, det = ±1;
    // partial-pivoting LU (faer's `MatRef::determinant`) is ample.
    let det = rotation.as_ref().determinant();
    if det < 0.0 {
        // Flip sign of the last column of U, then recompute R = U' · Vᵀ.
        // Materialise U into an owned Mat so we can mutate the column.
        let last = k - 1;
        let u_flipped = Mat::<f64>::from_fn(k, k, |i, j| {
            if j == last { -u[(i, j)] } else { u[(i, j)] }
        });
        matmul(
            rotation.as_mut(),
            Accum::Replace,
            u_flipped.as_ref(),
            v.transpose(),
            1.0,
            Par::Seq,
        );
    }

    // Recompute scale = trace(M · Rᵀ) = Σ_{i,j} M[i,j] · R[i,j] on the
    // possibly-flipped R. The cached identity ‖a·R − ref‖² = ‖a‖² + ‖ref‖² − 2·scale
    // holds for *any* orthogonal R, so OrthogonalAlignment::residual_frobenius
    // remains valid.
    let mut scale = 0.0;
    for i in 0..k {
        for j in 0..k {
            scale += m_buf[(i, j)] * rotation[(i, j)];
        }
    }

    Ok(OrthogonalAlignment { rotation, scale })
}

/// Result of [`orthogonal`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct OrthogonalAlignment {
    /// `K×K` orthogonal rotation `R` such that `a · R ≈ reference`.
    pub rotation: Mat<f64>,
    /// Nuclear norm `‖aᵀ · reference‖_*` — sum of singular values.
    /// Returned for parity with `SciPy`'s `orthogonal_procrustes`. Free
    /// byproduct of the SVD path. Use [`Self::residual_frobenius`] for the
    /// Frobenius distance, which costs `O(M·K)` per matrix and is therefore
    /// not eager.
    pub scale: f64,
}

impl OrthogonalAlignment {
    /// Compute `‖a · rotation − reference‖_F` for the same `(a, reference)`
    /// passed to [`orthogonal`]. Costs `O(M·K)` per matrix.
    #[must_use]
    pub fn residual_frobenius(&self, a: MatRef<'_, f64>, reference: MatRef<'_, f64>) -> f64 {
        // ‖aR − ref‖² = ‖a‖² + ‖ref‖² − 2·scale.
        let a_sq = frobenius_sq(a);
        let r_sq = frobenius_sq(reference);
        (a_sq + r_sq - 2.0 * self.scale).max(0.0).sqrt()
    }
}

fn frobenius_sq(x: MatRef<'_, f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            let v = x[(i, j)];
            s += v * v;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProcrustesError;
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat, Par};

    #[test]
    fn procrustes_recovers_known_rotation() {
        // a = reference @ R0; orthogonal should return R0ᵀ.
        let reference = Mat::<f64>::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 1.0,
            (2, 0) | (3, 1) => 0.5,
            _ => 0.0,
        });
        let theta = std::f64::consts::PI / 6.0;
        let r0 = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => theta.cos(),
            (0, 1) => -theta.sin(),
            (1, 0) => theta.sin(),
            _ => unreachable!(),
        });
        let a: Mat<f64> = &reference * &r0;

        let aln = orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();
        let recovered: Mat<f64> = &a * &aln.rotation;
        for i in 0..4 {
            for j in 0..2 {
                assert!(
                    (recovered[(i, j)] - reference[(i, j)]).abs() < 1e-10,
                    "i={i} j={j} got {} want {}",
                    recovered[(i, j)],
                    reference[(i, j)]
                );
            }
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn orthogonality_of_R() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let a = Mat::<f64>::from_fn(8, 4, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let reference = Mat::<f64>::from_fn(8, 4, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let aln = orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();
        let mut rtr = Mat::<f64>::zeros(4, 4);
        matmul(
            rtr.as_mut(),
            Accum::Replace,
            aln.rotation.transpose(),
            aln.rotation.as_ref(),
            1.0,
            Par::Seq,
        );
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (rtr[(i, j)] - want).abs() < 1e-12,
                    "RᵀR[{i},{j}] = {}",
                    rtr[(i, j)]
                );
            }
        }
    }

    #[test]
    fn procrustes_zero_input_returns_orthogonal() {
        let w = Mat::<f64>::zeros(5, 3);
        let aln = orthogonal(w.as_ref(), w.as_ref(), false).unwrap();
        let mut rtr = Mat::<f64>::zeros(3, 3);
        matmul(
            rtr.as_mut(),
            Accum::Replace,
            aln.rotation.transpose(),
            aln.rotation.as_ref(),
            1.0,
            Par::Seq,
        );
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((rtr[(i, j)] - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn scale_matches_nuclear_norm() {
        // For a = reference, scale = ‖aᵀa‖_* = sum of singular values of aᵀa.
        // For an orthonormal a (a = first K columns of identity), aᵀa = I_K so scale = K.
        let a = Mat::<f64>::from_fn(6, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let aln = orthogonal(a.as_ref(), a.as_ref(), false).unwrap();
        assert!(
            (aln.scale - 3.0).abs() < 1e-12,
            "scale = {} want 3",
            aln.scale
        );
    }

    #[test]
    fn residual_frobenius_method_matches_direct_computation() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let a = Mat::<f64>::from_fn(10, 3, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let reference =
            Mat::<f64>::from_fn(10, 3, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let aln = orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();

        // Direct: ‖a · R − reference‖_F.
        let mut ar = Mat::<f64>::zeros(10, 3);
        matmul(
            ar.as_mut(),
            Accum::Replace,
            a.as_ref(),
            aln.rotation.as_ref(),
            1.0,
            Par::Seq,
        );
        let mut direct_sq = 0.0;
        for i in 0..10 {
            for j in 0..3 {
                let d = ar[(i, j)] - reference[(i, j)];
                direct_sq += d * d;
            }
        }
        let direct = direct_sq.sqrt();

        let via_method = aln.residual_frobenius(a.as_ref(), reference.as_ref());
        assert!(
            (via_method - direct).abs() < 1e-12,
            "method {via_method} direct {direct}"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn k_eq_1() {
        let a = Mat::<f64>::from_fn(5, 1, |i, _| (i as f64) - 2.0);
        let reference = Mat::<f64>::from_fn(5, 1, |i, _| -((i as f64) - 2.0));
        let aln = orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();
        // K=1: only sign flip is available; expect rotation = [[-1.0]].
        assert!((aln.rotation[(0, 0)] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_input_returns_error() {
        let zero_rows = Mat::<f64>::zeros(0, 3);
        let zero_cols = Mat::<f64>::zeros(5, 0);
        assert!(matches!(
            orthogonal(zero_rows.as_ref(), zero_rows.as_ref(), false),
            Err(ProcrustesError::EmptyInput)
        ));
        assert!(matches!(
            orthogonal(zero_cols.as_ref(), zero_cols.as_ref(), false),
            Err(ProcrustesError::EmptyInput)
        ));
    }

    #[test]
    fn dim_mismatch_returns_error() {
        let a = Mat::<f64>::zeros(5, 3);
        let ref_rows = Mat::<f64>::zeros(4, 3);
        let ref_cols = Mat::<f64>::zeros(5, 2);
        assert!(matches!(
            orthogonal(a.as_ref(), ref_rows.as_ref(), false),
            Err(ProcrustesError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            orthogonal(a.as_ref(), ref_cols.as_ref(), false),
            Err(ProcrustesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn nan_with_check_finite_true_returns_error() {
        let mut a = Mat::<f64>::zeros(3, 2);
        a[(1, 1)] = f64::NAN;
        let reference = Mat::<f64>::zeros(3, 2);
        assert!(matches!(
            orthogonal(a.as_ref(), reference.as_ref(), true),
            Err(ProcrustesError::NonFinite)
        ));
    }

    #[test]
    fn nan_with_check_finite_false_does_not_panic() {
        let mut a = Mat::<f64>::zeros(3, 2);
        a[(1, 1)] = f64::NAN;
        let reference = Mat::<f64>::zeros(3, 2);
        // Result is undefined (NaNs propagate); we only assert no panic.
        let _ = orthogonal(a.as_ref(), reference.as_ref(), false);
    }
}
