//! Signed-permutation alignment.

use faer::MatRef;

use crate::{is_all_finite, ProcrustesError};

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

const BRUTE_FORCE_CUTOFF: usize = 3;

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
    // Heap's algorithm, textbook variant: one initial recurse, then n−1
    // (swap, recurse) pairs. Equivalent enumeration order to a `0..n` loop
    // that recurses-then-swaps, but without the wasted trailing swap. Generic
    // over `F` so the leaf score loop inlines through the recursion.
    fn heap_permute<F: FnMut(&[usize])>(buf: &mut [usize], n: usize, on_perm: &mut F) {
        if n == 1 {
            on_perm(buf);
            return;
        }
        heap_permute(buf, n - 1, on_perm);
        for i in 0..n - 1 {
            if n % 2 == 0 {
                buf.swap(i, n - 1);
            } else {
                buf.swap(0, n - 1);
            }
            heap_permute(buf, n - 1, on_perm);
        }
    }

    let mut perm: Vec<usize> = (0..k).collect();
    let mut best_assigned: Vec<usize> = perm.clone();
    let mut best_score = f64::NEG_INFINITY;

    // Precomputing `|dot|` would save K abs-ops per leaf but the K² Vec
    // allocation cost dominates at the production cutoff (K ≤ 3, 6 perms);
    // measured net-slower at every K ≤ 10. Inline the abs() instead.
    let mut on_perm = |p: &[usize]| {
        let mut score = 0.0;
        for kk in 0..k {
            score += dot[p[kk] * k + kk].abs();
        }
        if score > best_score {
            best_score = score;
            best_assigned.copy_from_slice(p);
        }
    };
    heap_permute(&mut perm, k, &mut on_perm);

    best_assigned
}

/// Solve `min_{P ∈ S_K, s ∈ {±1}^K} ‖a · P · diag(s) − reference‖_F` exactly.
///
/// # Algorithm
///
/// For `K ≤ 3`: brute-force enumeration of `K!` permutations
/// (cost-per-cell `nb[p[k]] − 2·|dot[p[k], k]| + nr[k]`, optimal sign per
/// permutation closed-form). For `K ≥ 4`: Jonker-Volgenant linear assignment
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

    #[allow(clippy::cast_precision_loss)]
    fn parity_check_brute_vs_jv(ks: &[usize]) {
        use rand::SeedableRng;

        for &k in ks {
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

    #[test]
    fn lap_matches_brute_force_at_k_9_and_10() {
        // K=9, 10 already cover the asymptotic JV-vs-brute parity invariant
        // and run in well under a second in debug mode. K=11 is split into
        // a separate `#[ignore]`d release-only test below.
        parity_check_brute_vs_jv(&[9_usize, 10]);
    }

    #[test]
    #[ignore = "slow: K=11 brute is ~1s release / ~60s debug; run with --release --ignored to verify"]
    fn lap_matches_brute_force_at_k_11() {
        parity_check_brute_vs_jv(&[11_usize]);
    }

    /// Maintainer-only calibration: time `brute_force_assign` and
    /// `lap::solve_max_abs` across `K ∈ [3, 12]` and recommend a
    /// `BRUTE_FORCE_CUTOFF` value for `src/signed_permutation.rs`.
    ///
    /// Run with:
    /// ```text
    /// cargo test --release --lib -- --ignored --nocapture calibrate_brute_force_cutoff
    /// ```
    ///
    /// Prints a per-K wall-time table plus a single `RECOMMENDATION:` line.
    /// Does **not** mutate the constant — read the output, hand-edit
    /// `BRUTE_FORCE_CUTOFF` above, and cite the run in the commit message.
    #[test]
    #[ignore = "maintainer-only: run with --release before each tag to set BRUTE_FORCE_CUTOFF"]
    #[allow(
        clippy::cast_precision_loss,
        clippy::many_single_char_names,
        clippy::too_many_lines,
        clippy::unreadable_literal
    )]
    fn calibrate_brute_force_cutoff() {
        use rand::SeedableRng;
        use std::time::Instant;

        // M = 64 is a midpoint sample size: large enough for cache effects
        // to look realistic, small enough to keep total wall-time under
        // ~30 s. K dominates wall-time at the brute-force end; M only
        // affects the up-front dot-matrix accumulation that both paths
        // share. Future maintainers can sweep M too if they care.
        const M: usize = 64;
        const SEED: u64 = 0xC0_FF_EE;
        // (K, iteration count). Counts target ≈ 100 ms per cell at small
        // K and ≤ ~15 s at K = 12 where one brute enumeration is ~5 s.
        const SCHEDULE: &[(usize, usize)] = &[
            (3, 1000),
            (4, 1000),
            (5, 1000),
            (6, 1000),
            (7, 1000),
            (8, 1000),
            (9, 500),
            (10, 100),
            (11, 21),
            (12, 3),
        ];

        fn median_us<F: FnMut()>(iters: usize, mut f: F) -> f64 {
            // One discarded warm-up iteration to seed branch predictor / caches.
            f();
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = Instant::now();
                f();
                times.push(t.elapsed().as_secs_f64() * 1e6);
            }
            times.sort_by(f64::total_cmp);
            times[times.len() / 2]
        }

        if cfg!(debug_assertions) {
            eprintln!(
                "WARNING: run with --release; debug-mode timings invert the brute/JV crossover."
            );
            return;
        }

        let host = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string());

        eprintln!("=== procrustes BRUTE_FORCE_CUTOFF calibration ===");
        eprintln!("host: {host}     opt-level: 3 (release)     M = {M}");
        eprintln!();
        eprintln!("  K    brute (µs)    jv (µs)    brute / jv");

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(SEED);
        let mut brute_wins = vec![false; SCHEDULE.len()];
        let mut recommend: usize = 1;

        for (idx, &(k, iters)) in SCHEDULE.iter().enumerate() {
            let a = Mat::<f64>::from_fn(M, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
            let reference =
                Mat::<f64>::from_fn(M, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));

            // Precompute dot once per K — both algorithms operate on the
            // same K×K buffer, so this matches signed_permutation's
            // dispatch site and isolates the comparison to the assignment
            // step itself.
            let mut dot = vec![0.0_f64; k * k];
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0;
                    for r in 0..M {
                        s += a[(r, i)] * reference[(r, j)];
                    }
                    dot[i * k + j] = s;
                }
            }

            let bf_us = median_us(iters, || {
                let _assigned = brute_force_assign(&dot, k);
            });
            let jv_us = median_us(iters, || {
                let _assigned = crate::lap::solve_max_abs(&dot, k);
            });

            let ratio = bf_us / jv_us;
            eprintln!("{k:>3}    {bf_us:>10.2}    {jv_us:>7.2}    {ratio:>10.2}");

            brute_wins[idx] = bf_us < jv_us; // strict: tie goes to JV
            if brute_wins[idx] {
                recommend = k;
            }
        }

        eprintln!();

        // Non-monotonicity: any K where brute wins after a previous K
        // where it lost. Asymptotically K!/K³ is monotonic for K ≥ 5, so
        // a non-monotonic stretch flags noisy small-iter cells (K = 11, 12).
        if let Some(first_loss) = brute_wins.iter().position(|&w| !w) {
            let bad: Vec<usize> = SCHEDULE[first_loss..]
                .iter()
                .enumerate()
                .filter_map(|(j, &(k, _))| brute_wins[first_loss + j].then_some(k))
                .collect();
            if !bad.is_empty() {
                eprintln!(
                    "WARNING: non-monotonic crossover at K ∈ {bad:?}; iteration counts may be too small."
                );
            }
        }

        let last_idx = SCHEDULE.len() - 1;
        if brute_wins[last_idx] {
            eprintln!("RECOMMENDATION: cutoff > 12; extend the sweep before applying.");
        } else if recommend == 1 {
            eprintln!("RECOMMENDATION: const BRUTE_FORCE_CUTOFF: usize = 1;");
            eprintln!(
                "WARNING: brute-force lost at every measured K; bit-parity tests assume the brute"
            );
            eprintln!("path is reachable via dispatch and may need updating.");
        } else {
            eprintln!("RECOMMENDATION: const BRUTE_FORCE_CUTOFF: usize = {recommend};");
        }

        eprintln!();
        eprintln!(
            "Reminder: parity tests at K ∈ {{9, 10}} (`lap_matches_brute_force_at_k_9_and_10`)"
        );
        eprintln!("and K=11 (`lap_matches_brute_force_at_k_11`, ignored by default) require the");
        eprintln!(
            "brute path to be callable directly via `brute_force_assign`. They do not depend"
        );
        eprintln!("on the dispatch constant — leave them as-is regardless of cutoff.");
    }
}
