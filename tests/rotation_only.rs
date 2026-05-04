//! Integration tests for `rotation_only`.

use procrustes::{orthogonal, rotation_only, Mat, MatRef, ProcrustesError};

/// Compute det of a small K×K matrix via faer's `MatRef::determinant`
/// (partial-pivot LU internally; ample for orthogonal `R` where `det = ±1`).
fn det(m: MatRef<'_, f64>) -> f64 {
    m.determinant()
}

#[test]
fn identity_input_returns_identity_with_det_plus_one() {
    let reference = Mat::<f64>::from_fn(4, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    let a = reference.clone();
    let aln = rotation_only(a.as_ref(), reference.as_ref(), true).unwrap();
    let d = det(aln.rotation.as_ref());
    assert!((d - 1.0).abs() < 1e-12, "det = {d}");
    let residual = aln.residual_frobenius(a.as_ref(), reference.as_ref());
    assert!(residual < 1e-10, "residual = {residual}");
}

#[test]
fn known_rotation_recovered_with_det_plus_one() {
    // R0 ∈ SO(2): rotation by π/4. Set a = reference · R0ᵀ; rotation_only
    // should recover R ≈ R0.
    let reference = Mat::<f64>::from_fn(6, 2, |i, j| match (i, j) {
        (0, 0) | (1, 1) => 1.0,
        (2, 0) | (3, 1) => 0.5,
        (4, 0) => 0.7,
        (5, 1) => -0.3,
        _ => 0.0,
    });
    let theta = std::f64::consts::PI / 4.0;
    let r0 = Mat::<f64>::from_fn(2, 2, |i, j| match (i, j) {
        (0, 0) | (1, 1) => theta.cos(),
        (0, 1) => -theta.sin(),
        (1, 0) => theta.sin(),
        _ => unreachable!(),
    });
    // a = reference · R0ᵀ → fitted R should be R0.
    let r0_t = Mat::<f64>::from_fn(2, 2, |i, j| r0[(j, i)]);
    let a: Mat<f64> = &reference * &r0_t;

    let aln = rotation_only(a.as_ref(), reference.as_ref(), false).unwrap();
    let d = det(aln.rotation.as_ref());
    assert!((d - 1.0).abs() < 1e-12, "det = {d}");
    for i in 0..2 {
        for j in 0..2 {
            let got = aln.rotation[(i, j)];
            let want = r0[(i, j)];
            assert!((got - want).abs() < 1e-10, "R[{i},{j}] = {got} want {want}");
        }
    }
}

#[test]
fn reflection_input_returns_proper_rotation_with_larger_residual() {
    // a = reference with column 0 negated. `orthogonal` returns a reflection
    // (det = -1) with residual ≈ 0; `rotation_only` returns the nearest
    // SO(K) rotation with strictly larger residual. Reference must be
    // full column rank so that `aᵀ·reference` has no zero singular values
    // and the SVD's column-sign assignment is determinate.
    let ref_data: [[f64; 3]; 5] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.4, 0.3, 0.2],
        [0.1, 0.5, 0.7],
    ];
    let reference = Mat::<f64>::from_fn(5, 3, |i, j| ref_data[i][j]);
    let a = Mat::<f64>::from_fn(5, 3, |i, j| {
        if j == 0 {
            -ref_data[i][j]
        } else {
            ref_data[i][j]
        }
    });

    let aln_o = orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();
    let aln_r = rotation_only(a.as_ref(), reference.as_ref(), false).unwrap();

    let det_o = det(aln_o.rotation.as_ref());
    let det_r = det(aln_r.rotation.as_ref());
    assert!(
        (det_o + 1.0).abs() < 1e-10,
        "orthogonal det = {det_o} want -1"
    );
    assert!(
        (det_r - 1.0).abs() < 1e-10,
        "rotation_only det = {det_r} want +1"
    );

    let res_o = aln_o.residual_frobenius(a.as_ref(), reference.as_ref());
    let res_r = aln_r.residual_frobenius(a.as_ref(), reference.as_ref());
    assert!(
        res_r > res_o + 1e-6,
        "rotation_only residual {res_r} should exceed orthogonal residual {res_o}"
    );
}

#[test]
fn det_is_plus_one_for_random_inputs_across_k() {
    use rand::SeedableRng;
    for k in [2_usize, 3, 4, 5, 8] {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x00C0_FFEE_u64 + k as u64);
        let a = Mat::<f64>::from_fn(16, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let reference =
            Mat::<f64>::from_fn(16, k, |_, _| rand::Rng::gen_range(&mut rng, -1.0..1.0));
        let aln = rotation_only(a.as_ref(), reference.as_ref(), false).unwrap();
        let d = det(aln.rotation.as_ref());
        assert!(
            (d - 1.0).abs() < 1e-12,
            "K={k}: det(R) = {d}, expected +1.0"
        );
    }
}

#[test]
fn shape_mismatch_returns_error() {
    let a = Mat::<f64>::zeros(5, 3);
    let bad_rows = Mat::<f64>::zeros(4, 3);
    let bad_cols = Mat::<f64>::zeros(5, 2);
    assert!(matches!(
        rotation_only(a.as_ref(), bad_rows.as_ref(), false),
        Err(ProcrustesError::DimensionMismatch { .. })
    ));
    assert!(matches!(
        rotation_only(a.as_ref(), bad_cols.as_ref(), false),
        Err(ProcrustesError::DimensionMismatch { .. })
    ));
}

#[test]
fn empty_input_returns_error() {
    let zero_rows = Mat::<f64>::zeros(0, 3);
    let zero_cols = Mat::<f64>::zeros(5, 0);
    assert!(matches!(
        rotation_only(zero_rows.as_ref(), zero_rows.as_ref(), false),
        Err(ProcrustesError::EmptyInput)
    ));
    assert!(matches!(
        rotation_only(zero_cols.as_ref(), zero_cols.as_ref(), false),
        Err(ProcrustesError::EmptyInput)
    ));
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn k_eq_1_always_returns_identity() {
    // SO(1) = {[[1.0]]}, so even when a sign flip would minimize residual,
    // rotation_only must return the identity rotation.
    let a = Mat::<f64>::from_fn(5, 1, |i, _| (i as f64) - 2.0);
    let reference = Mat::<f64>::from_fn(5, 1, |i, _| -((i as f64) - 2.0));
    let aln = rotation_only(a.as_ref(), reference.as_ref(), false).unwrap();
    assert!(
        (aln.rotation[(0, 0)] - 1.0).abs() < 1e-12,
        "K=1 rotation_only must return [[1.0]], got [[{}]]",
        aln.rotation[(0, 0)]
    );
    let d = det(aln.rotation.as_ref());
    assert!((d - 1.0).abs() < 1e-12, "det = {d}");
}

#[test]
fn nan_with_check_finite_true_returns_error() {
    let mut a = Mat::<f64>::zeros(3, 2);
    a[(1, 1)] = f64::NAN;
    let reference = Mat::<f64>::zeros(3, 2);
    assert!(matches!(
        rotation_only(a.as_ref(), reference.as_ref(), true),
        Err(ProcrustesError::NonFinite)
    ));
}
