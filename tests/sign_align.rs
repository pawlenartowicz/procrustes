//! Integration tests for `sign_align`.

use procrustes::{sign_align, Mat, ProcrustesError};

#[test]
#[allow(clippy::cast_precision_loss)]
fn identity_input_all_plus_one() {
    // a == reference → every sign is +1.0, residual is zero.
    let reference = Mat::<f64>::from_fn(5, 3, |i, j| (i as f64) * 0.7 + (j as f64));
    let a = reference.clone();
    let aln = sign_align(a.as_ref(), reference.as_ref(), true).unwrap();
    assert_eq!(aln.signs, vec![1.0, 1.0, 1.0]);
    assert!(
        aln.residual_frobenius < 1e-12,
        "residual_frobenius = {}",
        aln.residual_frobenius
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn one_column_flipped() {
    // a is reference with column 1 negated → signs = [+1, -1, +1].
    let reference = Mat::<f64>::from_fn(5, 3, |i, j| (i as f64) * 0.7 + (j as f64) + 1.0);
    let a = Mat::<f64>::from_fn(5, 3, |i, j| {
        let x = (i as f64) * 0.7 + (j as f64) + 1.0;
        if j == 1 {
            -x
        } else {
            x
        }
    });
    let aln = sign_align(a.as_ref(), reference.as_ref(), true).unwrap();
    assert_eq!(aln.signs, vec![1.0, -1.0, 1.0]);
    assert!(
        aln.residual_frobenius < 1e-12,
        "residual_frobenius = {}",
        aln.residual_frobenius
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn full_negation_all_minus_one() {
    let reference = Mat::<f64>::from_fn(4, 2, |i, j| (i as f64) + (j as f64) * 2.0 + 0.3);
    let a = Mat::<f64>::from_fn(4, 2, |i, j| -((i as f64) + (j as f64) * 2.0 + 0.3));
    let aln = sign_align(a.as_ref(), reference.as_ref(), true).unwrap();
    assert_eq!(aln.signs, vec![-1.0, -1.0]);
    assert!(
        aln.residual_frobenius < 1e-12,
        "residual_frobenius = {}",
        aln.residual_frobenius
    );
}

#[test]
fn shape_mismatch_returns_error() {
    let a = Mat::<f64>::zeros(5, 3);
    let bad_rows = Mat::<f64>::zeros(4, 3);
    let bad_cols = Mat::<f64>::zeros(5, 2);
    assert!(matches!(
        sign_align(a.as_ref(), bad_rows.as_ref(), false),
        Err(ProcrustesError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        sign_align(a.as_ref(), bad_cols.as_ref(), false),
        Err(ProcrustesError::DimensionMismatch { .. })
    ));
}

#[test]
fn empty_input_returns_error() {
    let zero_rows = Mat::<f64>::zeros(0, 3);
    let zero_cols = Mat::<f64>::zeros(5, 0);
    assert!(matches!(
        sign_align(zero_rows.as_ref(), zero_rows.as_ref(), false),
        Err(ProcrustesError::EmptyInput)
    ));
    assert!(matches!(
        sign_align(zero_cols.as_ref(), zero_cols.as_ref(), false),
        Err(ProcrustesError::EmptyInput)
    ));
}

#[test]
fn nan_with_check_finite_true_returns_error() {
    let mut a = Mat::<f64>::zeros(3, 2);
    a[(1, 1)] = f64::NAN;
    let reference = Mat::<f64>::zeros(3, 2);
    assert!(matches!(
        sign_align(a.as_ref(), reference.as_ref(), true),
        Err(ProcrustesError::NonFinite)
    ));
}

#[test]
fn zero_dot_product_tie_breaks_to_plus_one() {
    // Construct columns where ⟨a[:,k], ref[:,k]⟩ = 0 exactly.
    // Take reference = [[1, 0], [0, 1]] (4x2 padded with zeros) and
    // a = [[0, 1], [1, 0]] (orthogonal columns to reference's columns).
    let reference = Mat::<f64>::from_fn(4, 2, |i, j| match (i, j) {
        (0, 0) | (1, 1) => 1.0,
        _ => 0.0,
    });
    let a = Mat::<f64>::from_fn(4, 2, |i, j| match (i, j) {
        (0, 1) | (1, 0) => 1.0,
        _ => 0.0,
    });
    // ⟨a[:,0], ref[:,0]⟩ = 0 and ⟨a[:,1], ref[:,1]⟩ = 0 — both tie.
    let aln = sign_align(a.as_ref(), reference.as_ref(), true).unwrap();
    assert_eq!(aln.signs, vec![1.0, 1.0]);
    // residual² = ‖a‖² + ‖ref‖² − 2·Σ|dot| = 2 + 2 − 0 = 4 → residual = 2.
    assert!((aln.residual_frobenius - 2.0).abs() < 1e-12);
}

#[test]
fn residual_matches_direct_computation() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let a = Mat::<f64>::from_fn(8, 4, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
    let reference = Mat::<f64>::from_fn(8, 4, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
    let aln = sign_align(a.as_ref(), reference.as_ref(), false).unwrap();

    // Direct: ‖a · diag(signs) − reference‖_F.
    let mut direct_sq = 0.0;
    for j in 0..4 {
        for i in 0..8 {
            let d = aln.signs[j] * a[(i, j)] - reference[(i, j)];
            direct_sq += d * d;
        }
    }
    let direct = direct_sq.sqrt();
    assert!(
        (aln.residual_frobenius - direct).abs() < 1e-12,
        "got {} want {}",
        aln.residual_frobenius,
        direct
    );
}
