//! Signed-permutation alignment.

use faer::MatRef;

use crate::ProcrustesError;

/// Result of [`sign_align`].
///
/// Mirrors the eager-`residual_frobenius` shape of
/// [`SignedPermutationAlignment`] for the degenerate identity-permutation
/// case.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SignAlignment {
    /// Length-`K` vector of `±1.0`; entry `k` is the sign applied to column
    /// `k` of `a`.
    pub signs: Vec<f64>,
    /// Frobenius distance `‖a · diag(signs) − reference‖_F`. Un-squared,
    /// matching [`SignedPermutationAlignment::residual_frobenius`]; callers
    /// who need the squared form take `r.powi(2)`.
    pub residual_frobenius: f64,
}

/// Sign-only alignment: for each column `k`, choose `s[k] ∈ {−1, +1}` to
/// maximise `⟨s[k] · a[:, k], reference[:, k]⟩`. Closed-form, `O(M·K)`.
///
/// Use this when columns of `a` are already in the same order as
/// `reference` and only the per-column sign is arbitrary — the canonical
/// PLS bootstrap pattern. For a general column-and-sign search, see
/// [`signed_permutation`].
///
/// Tie-breaking: a column dot-product of exactly `0.0` returns `+1.0`.
///
/// # Errors
/// Same as [`crate::orthogonal`]:
/// [`ProcrustesError::DimensionMismatch`] on shape mismatch,
/// [`ProcrustesError::EmptyInput`] if either dimension is zero,
/// [`ProcrustesError::NonFinite`] if `check_finite` is `true` and any input
/// value is NaN or infinite.
///
/// # Examples
/// ```
/// use procrustes::Mat;
/// let reference = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
/// // a is reference with column 1 sign-flipped.
/// let a = Mat::<f64>::from_fn(4, 2, |i, j| {
///     if i == j {
///         if j == 1 { -1.0 } else { 1.0 }
///     } else {
///         0.0
///     }
/// });
/// let aln = procrustes::sign_align(a.as_ref(), reference.as_ref(), true).unwrap();
/// assert_eq!(aln.signs, vec![1.0, -1.0]);
/// assert!(aln.residual_frobenius < 1e-12);
/// ```
pub fn sign_align(
    a: MatRef<'_, f64>,
    reference: MatRef<'_, f64>,
    check_finite: bool,
) -> Result<SignAlignment, ProcrustesError> {
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

    let m = a_rows;
    let k = a_cols;

    let mut signs = Vec::with_capacity(k);
    let mut residual_sq = 0.0;
    for col in 0..k {
        let mut dot = 0.0;
        let mut a_norm_sq = 0.0;
        let mut ref_norm_sq = 0.0;
        for row in 0..m {
            let av = a[(row, col)];
            let rv = reference[(row, col)];
            dot += av * rv;
            a_norm_sq += av * av;
            ref_norm_sq += rv * rv;
        }
        // Tie-breaking: dot exactly 0.0 → +1.0 (documented).
        let s = if dot >= 0.0 { 1.0_f64 } else { -1.0_f64 };
        signs.push(s);
        // ‖a·diag(s) − ref‖² per column = ‖a‖² − 2·s·dot + ‖ref‖²
        //                                = ‖a‖² − 2·|dot| + ‖ref‖² (since s·dot = |dot|).
        residual_sq += a_norm_sq - 2.0 * dot.abs() + ref_norm_sq;
    }

    Ok(SignAlignment {
        signs,
        residual_frobenius: residual_sq.max(0.0).sqrt(),
    })
}

const BRUTE_FORCE_CUTOFF: usize = 8;

/// Brute-force max-|dot| assignment by `K!` permutation enumeration.
///
/// Returns `assigned` where `assigned[k]` = source-column-of-`a` mapped to
/// column `k` of `reference`. Uses the score form `Σ_k |dot[p[k], k]|`
/// as the criterion (equivalent to minimising the Frobenius cost but without
/// needing `nb`/`nr`). Caller derives signs and residual from `assigned`.
/// Used directly for `K ≤ BRUTE_FORCE_CUTOFF`; exposed `pub(crate)` so the
/// parity unit test inside this module (Task 3) can call it directly.
#[allow(clippy::many_single_char_names)]
pub(crate) fn brute_force_assign(dot: &[f64], k: usize) -> Vec<usize> {
    fn heap_permute(buf: &mut Vec<usize>, n: usize, on_perm: &mut dyn FnMut(&[usize])) {
        if n == 1 {
            on_perm(buf);
            return;
        }
        for i in 0..n {
            heap_permute(buf, n - 1, on_perm);
            if n % 2 == 0 {
                buf.swap(i, n - 1);
            } else {
                buf.swap(0, n - 1);
            }
        }
    }

    let mut perm: Vec<usize> = (0..k).collect();
    let mut best_assigned: Vec<usize> = perm.clone();
    let mut best_score = f64::NEG_INFINITY;

    let mut on_perm = |p: &[usize]| {
        let mut score = 0.0;
        for kk in 0..k {
            score += dot[p[kk] * k + kk].abs();
        }
        if score > best_score {
            best_score = score;
            best_assigned.clear();
            best_assigned.extend_from_slice(p);
        }
    };
    heap_permute(&mut perm, k, &mut on_perm);

    best_assigned
}

/// Solve `min_{P ∈ S_K, s ∈ {±1}^K} ‖a · P · diag(s) − reference‖_F` exactly.
///
/// # Algorithm
///
/// For `K ≤ 8`: brute-force enumeration of `K!` permutations
/// (cost-per-cell `nb[p[k]] − 2·|dot[p[k], k]| + nr[k]`, optimal sign per
/// permutation closed-form). For `K ≥ 9`: Jonker-Volgenant linear assignment
/// on `-|aᵀ·reference|`, `O(K³)`. Both paths return the global optimum;
/// **permutation at exact cost ties is implementation-defined** and may
/// differ between the two paths.
///
/// # Errors
/// Same as [`crate::orthogonal`].
///
/// # Examples
/// ```
/// use procrustes::Mat;
/// // a is the 4×2 identity columns swapped, with column 1 sign-flipped.
/// let a = Mat::<f64>::from_fn(4, 2, |i, j| match (i, j) {
///     (0, 1) => 1.0,
///     (1, 0) => -1.0,
///     _ => 0.0,
/// });
/// let reference = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.0 });
/// let aln = procrustes::signed_permutation(a.as_ref(), reference.as_ref(), true).unwrap();
/// assert_eq!(aln.assigned, vec![1, 0]);
/// assert_eq!(aln.signs, vec![1.0, -1.0]);
/// assert!(aln.residual_frobenius < 1e-10);
/// ```
#[allow(clippy::many_single_char_names)]
pub fn signed_permutation(
    a: MatRef<'_, f64>,
    reference: MatRef<'_, f64>,
    check_finite: bool,
) -> Result<SignedPermutationAlignment, ProcrustesError> {
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

    let d = a_rows;
    let k = a_cols;

    // dot[i, j] = ⟨a[:, i], reference[:, j]⟩
    let mut dot = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for r in 0..d {
                s += a[(r, i)] * reference[(r, j)];
            }
            dot[i * k + j] = s;
        }
    }

    // ‖a[:, i]‖² and ‖reference[:, j]‖²
    let mut nb = vec![0.0_f64; k];
    let mut nr = vec![0.0_f64; k];
    for i in 0..k {
        let mut sb = 0.0;
        let mut sr = 0.0;
        for r in 0..d {
            sb += a[(r, i)] * a[(r, i)];
            sr += reference[(r, i)] * reference[(r, i)];
        }
        nb[i] = sb;
        nr[i] = sr;
    }

    let assigned = if k <= BRUTE_FORCE_CUTOFF {
        brute_force_assign(&dot, k)
    } else {
        crate::lap::solve_max_abs(&dot, k)
    };

    // Compute signs and cost in a single for-loop accumulator (same operand
    // order and operations as the original closure, preserving bit parity).
    let mut cost = 0.0;
    let mut signs = vec![0.0_f64; k];
    for kk in 0..k {
        let d_pk = dot[assigned[kk] * k + kk];
        cost += nb[assigned[kk]] - 2.0 * d_pk.abs() + nr[kk];
        signs[kk] = if d_pk >= 0.0 { 1.0 } else { -1.0 };
    }
    let residual_frobenius = cost.max(0.0).sqrt();

    Ok(SignedPermutationAlignment {
        assigned,
        signs,
        residual_frobenius,
    })
}

fn is_all_finite(x: MatRef<'_, f64>) -> bool {
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            if !x[(i, j)].is_finite() {
                return false;
            }
        }
    }
    true
}

/// Result of [`signed_permutation`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SignedPermutationAlignment {
    /// Length-K permutation: `assigned[k]` is the source column of `a`
    /// mapped onto column `k` of `reference`.
    pub assigned: Vec<usize>,
    /// Length-K signs (each ±1.0) applied to the permuted columns of `a`.
    pub signs: Vec<f64>,
    /// Frobenius distance `‖a · P · diag(signs) − reference‖_F` where
    /// `P` is the permutation matrix encoded by [`Self::assigned`];
    /// equivalently, `sqrt(Σ_k ‖signs[k]·a[:, assigned[k]] − reference[:, k]‖²)`.
    pub residual_frobenius: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProcrustesError;
    use faer::Mat;

    #[test]
    fn signed_perm_recovers_swap_with_sign_flip() {
        // reference columns: c0 = (1, 0, 0)ᵀ, c1 = (0, 1, 0)ᵀ.
        // a has c0 ↔ c1 swapped and one column sign-flipped.
        let reference = Mat::<f64>::from_fn(3, 2, |i, j| match (i, j) {
            (0, 0) | (1, 1) => 1.0,
            _ => 0.0,
        });
        let a = Mat::<f64>::from_fn(3, 2, |i, j| match (i, j) {
            (0, 1) => 1.0,
            (1, 0) => -1.0,
            _ => 0.0,
        });
        let out = signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap();
        assert_eq!(out.assigned, vec![1, 0]);
        assert_eq!(out.signs, vec![1.0, -1.0]);
        assert!(
            out.residual_frobenius < 1e-10,
            "got {}",
            out.residual_frobenius
        );
    }

    #[test]
    fn signed_perm_identity_when_already_aligned() {
        let w = Mat::<f64>::from_fn(4, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let out = signed_permutation(w.as_ref(), w.as_ref(), false).unwrap();
        assert_eq!(out.assigned, vec![0, 1, 2]);
        assert_eq!(out.signs, vec![1.0, 1.0, 1.0]);
        assert!(out.residual_frobenius < 1e-12);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn k_eq_1_trivial() {
        // a = -reference (single column); expect assigned=[0], signs=[-1.0].
        let reference = Mat::<f64>::from_fn(5, 1, |i, _| (i as f64) - 2.0);
        let a = Mat::<f64>::from_fn(5, 1, |i, _| -((i as f64) - 2.0));
        let out = signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap();
        assert_eq!(out.assigned, vec![0]);
        assert_eq!(out.signs, vec![-1.0]);
        assert!(out.residual_frobenius < 1e-12);
    }

    #[test]
    fn k_eq_8_recovers_known_alignment() {
        // Build random a (K=8), construct reference by applying a known signed
        // permutation, and verify recovery.
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(123);
        let m = 24;
        let k = 8;
        let a = Mat::<f64>::from_fn(m, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));

        let true_perm: Vec<usize> = vec![3, 0, 7, 1, 5, 2, 6, 4];
        let true_signs: Vec<f64> = vec![1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

        // reference[:, k_out] = signs[k_out] · a[:, perm[k_out]]
        let reference = Mat::<f64>::from_fn(m, k, |i, j| true_signs[j] * a[(i, true_perm[j])]);

        let out = signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap();
        assert_eq!(out.assigned, true_perm);
        assert_eq!(out.signs, true_signs);
        assert!(
            out.residual_frobenius < 1e-10,
            "got {}",
            out.residual_frobenius
        );
    }

    #[test]
    fn empty_input_returns_error() {
        let z = Mat::<f64>::zeros(0, 3);
        assert!(matches!(
            signed_permutation(z.as_ref(), z.as_ref(), false),
            Err(ProcrustesError::EmptyInput)
        ));
        let zc = Mat::<f64>::zeros(5, 0);
        assert!(matches!(
            signed_permutation(zc.as_ref(), zc.as_ref(), false),
            Err(ProcrustesError::EmptyInput)
        ));
    }

    #[test]
    fn dim_mismatch_returns_error() {
        let a = Mat::<f64>::zeros(5, 3);
        let r1 = Mat::<f64>::zeros(4, 3);
        let r2 = Mat::<f64>::zeros(5, 2);
        assert!(matches!(
            signed_permutation(a.as_ref(), r1.as_ref(), false),
            Err(ProcrustesError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            signed_permutation(a.as_ref(), r2.as_ref(), false),
            Err(ProcrustesError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn nan_with_check_finite_true_returns_error() {
        let mut a = Mat::<f64>::zeros(3, 2);
        a[(1, 0)] = f64::NAN;
        let reference = Mat::<f64>::zeros(3, 2);
        assert!(matches!(
            signed_permutation(a.as_ref(), reference.as_ref(), true),
            Err(ProcrustesError::NonFinite)
        ));
    }

    #[test]
    fn nan_with_check_finite_false_does_not_panic() {
        let mut a = Mat::<f64>::zeros(3, 2);
        a[(1, 0)] = f64::NAN;
        let reference = Mat::<f64>::zeros(3, 2);
        let _ = signed_permutation(a.as_ref(), reference.as_ref(), false);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn lap_matches_brute_force_at_k_9_through_11() {
        use rand::SeedableRng;

        for &k in &[9_usize, 10, 11] {
            for seed in 0_u64..5 {
                let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                let m = 32;
                let a = Mat::<f64>::from_fn(m, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
                let reference =
                    Mat::<f64>::from_fn(m, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));

                // Replicate the dot-precompute from signed_permutation (same arithmetic, same order).
                let mut dot = vec![0.0_f64; k * k];
                for i in 0..k {
                    for j in 0..k {
                        let mut s = 0.0;
                        for r in 0..m {
                            s += a[(r, i)] * reference[(r, j)];
                        }
                        dot[i * k + j] = s;
                    }
                }

                let bf_assigned = brute_force_assign(&dot, k);
                let jv_assigned = crate::lap::solve_max_abs(&dot, k);

                // Compute Σ_k |dot[p[k], k]| for both — equal optima ⇒ equal cost.
                let bf_score: f64 = (0..k).map(|kk| dot[bf_assigned[kk] * k + kk].abs()).sum();
                let jv_score: f64 = (0..k).map(|kk| dot[jv_assigned[kk] * k + kk].abs()).sum();

                assert!(
                    (bf_score - jv_score).abs() < 1e-12,
                    "K={k} seed={seed}: bf score {bf_score} vs jv score {jv_score}",
                );
            }
        }
    }
}
