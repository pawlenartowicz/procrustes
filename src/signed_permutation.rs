//! Brute-force signed-permutation alignment.

use faer::MatRef;

use crate::ProcrustesError;

/// Solve `min_{P ∈ S_K, s ∈ {±1}^K} ‖a · P · diag(s) − reference‖_F` by
/// brute-force enumeration of `K!` permutations (optimal sign per
/// permutation is closed-form: `s[k] = sign(⟨a[:, p[k]], reference[:, k]⟩)`).
///
/// # Runtime
///
/// Per-call cost is `O(K! · K)` permutation evaluations after an
/// `O(K² · M)` precompute of the `aᵀ · reference` dot-product matrix.
/// Factorial growth means the practical ceiling is roughly `K ≤ 10` for
/// interactive use; concrete wall-times depend on hardware and compiler
/// (see the `print_runtime_table` test in this module). For `K ≳ 12`,
/// prefer manual decomposition.
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

    // Brute-force enumerate K! permutations. perm[k_out] names the source
    // column of `a` that maps onto reference[:, k_out]. Optimal sign per
    // output column k_out is sign(dot[perm[k_out], k_out]); cost-per-k
    // simplifies to nb[perm[k]] − 2·|dot[perm[k], k]| + nr[k].
    let mut perm: Vec<usize> = (0..k).collect();
    let mut best_assigned: Vec<usize> = perm.clone();
    let mut best_signs: Vec<f64> = vec![1.0; k];
    let mut best_cost = f64::INFINITY;
    let mut signs_scratch = vec![0.0_f64; k];

    let mut on_perm = |p: &[usize]| {
        let mut cost = 0.0;
        for kk in 0..k {
            let d_pk = dot[p[kk] * k + kk];
            cost += nb[p[kk]] - 2.0 * d_pk.abs() + nr[kk];
            signs_scratch[kk] = if d_pk >= 0.0 { 1.0 } else { -1.0 };
        }
        if cost < best_cost {
            best_cost = cost;
            best_assigned.clear();
            best_assigned.extend_from_slice(p);
            best_signs.clone_from(&signs_scratch);
        }
    };
    heap_permute(&mut perm, k, &mut on_perm);

    let residual_frobenius = best_cost.max(0.0).sqrt();
    Ok(SignedPermutationAlignment {
        assigned: best_assigned,
        signs: best_signs,
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

    /// Informational: prints wall-times for K ∈ {4, 6, 8, 10}. Skipped if
    /// `PROCRUSTES_SKIP_TIMING` is set. No assertions — populates docs.
    #[test]
    fn print_runtime_table() {
        use rand::SeedableRng;
        if std::env::var_os("PROCRUSTES_SKIP_TIMING").is_some() {
            return;
        }
        for &k in &[4_usize, 6, 8, 10] {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x00C0_FFEE);
            let m = 32;
            let a = Mat::<f64>::from_fn(m, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
            let reference = a.clone();
            let start = std::time::Instant::now();
            let _ = signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap();
            let dur = start.elapsed();
            eprintln!("signed_permutation K={k}: {dur:?}");
        }
    }
}
