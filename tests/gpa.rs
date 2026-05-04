//! Functional tests for [`procrustes::generalized`].

use procrustes::{generalized, GpaInit, GpaOptions, InnerAligner, Mat, MatRef, ProcrustesError};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a random `M × K` matrix from a seeded RNG.
fn random_mat(m: usize, k: usize, rng: &mut rand_chacha::ChaCha8Rng) -> Mat<f64> {
    Mat::<f64>::from_fn(m, k, |_, _| rand::Rng::gen_range(rng, -1.0..1.0))
}

/// Build a random `K × K` orthogonal matrix via Gram–Schmidt on a random
/// matrix with entries in `[-1, 1]`.  Uses the seeded `rng`.
fn random_orthogonal(k: usize, rng: &mut rand_chacha::ChaCha8Rng) -> Mat<f64> {
    // Build K column-vectors, each length K, orthonormalised via
    // modified Gram–Schmidt.
    let raw: Vec<Vec<f64>> = (0..k)
        .map(|_| {
            (0..k)
                .map(|_| rand::Rng::gen_range(rng, -1.0_f64..1.0))
                .collect()
        })
        .collect();

    let mut q: Vec<Vec<f64>> = Vec::with_capacity(k);
    for col in raw {
        let mut v = col;
        for qj in &q {
            let dot: f64 = v.iter().zip(qj.iter()).map(|(a, b)| a * b).sum();
            for (vi, &qi) in v.iter_mut().zip(qj.iter()) {
                *vi -= dot * qi;
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for vi in &mut v {
            *vi /= norm;
        }
        q.push(v);
    }

    // q[j][i] = Q[i, j]  (column j, row i).
    Mat::<f64>::from_fn(k, k, |i, j| q[j][i])
}

/// Maximum pairwise Frobenius distance among all pairs in `aligned`.
fn max_pair_diff(aligned: &[Mat<f64>]) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..aligned.len() {
        for j in (i + 1)..aligned.len() {
            let m = aligned[i].nrows();
            let k = aligned[i].ncols();
            let mut s = 0.0;
            for r in 0..m {
                for c in 0..k {
                    let d = aligned[i][(r, c)] - aligned[j][(r, c)];
                    s += d * d;
                }
            }
            worst = worst.max(s.sqrt());
        }
    }
    worst
}

/// Frobenius norm of a matrix.
fn frobenius(m: &Mat<f64>) -> f64 {
    let mut s = 0.0;
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let v = m[(i, j)];
            s += v * v;
        }
    }
    s.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn identical_inputs_converge_in_one_iteration() {
    let a = Mat::<f64>::from_fn(5, 3, |i, j| if i == j { 1.0 } else { 0.0 });
    let mats = [a.as_ref(), a.as_ref(), a.as_ref()];

    let aln = generalized(&mats, GpaOptions::default()).unwrap();

    assert!(aln.converged, "should have converged");
    assert_eq!(aln.n_iters, 1, "should converge in one iteration");
    for i in 0..5 {
        for j in 0..3 {
            let diff = (aln.consensus[(i, j)] - a[(i, j)]).abs();
            assert!(diff < 1e-12, "consensus[{i},{j}] diff {diff}");
        }
    }
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn single_matrix_n_eq_1() {
    let a = Mat::<f64>::from_fn(4, 2, |i, j| ((i + j) as f64) * 0.5);
    let mats = [a.as_ref()];

    let aln = generalized(&mats, GpaOptions::default()).unwrap();

    assert!(aln.converged);
    assert_eq!(aln.n_iters, 1);
    assert_eq!(aln.aligned.len(), 1);
    // consensus should equal a (aligning a to itself is identity).
    for i in 0..4 {
        for j in 0..2 {
            let diff = (aln.consensus[(i, j)] - a[(i, j)]).abs();
            assert!(diff < 1e-12, "consensus[{i},{j}] diff {diff}");
        }
    }
}

#[test]
fn random_orthogonal_perturbed_inputs_orthogonal_aligner() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

    let m = 8;
    let k = 3;
    let base = random_mat(m, k, &mut rng);

    // Build A_i = base · R_i for 5 random orthogonal R_i.
    let inputs_owned: Vec<Mat<f64>> = (0..5)
        .map(|_| {
            let r = random_orthogonal(k, &mut rng);
            Mat::<f64>::from_fn(m, k, |i, j| {
                let mut v = 0.0;
                for l in 0..k {
                    v += base[(i, l)] * r[(l, j)];
                }
                v
            })
        })
        .collect();
    let mat_refs: Vec<MatRef<'_, f64>> = inputs_owned.iter().map(Mat::as_ref).collect();

    let aln = generalized(
        &mat_refs,
        GpaOptions {
            inner: InnerAligner::Orthogonal,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(aln.converged, "should have converged");
    assert!(aln.n_iters <= 20, "n_iters = {}", aln.n_iters);

    let diff = max_pair_diff(&aln.aligned);
    assert!(diff < 1e-8, "max pairwise diff = {diff}");

    // Consensus equals base up to a global rotation — verify via one
    // additional orthogonal alignment.
    let final_aln =
        procrustes::orthogonal(aln.consensus.as_ref(), base.as_ref(), false).unwrap();
    let residual = final_aln.residual_frobenius(aln.consensus.as_ref(), base.as_ref());
    assert!(residual < 1e-8, "consensus-vs-base residual = {residual}");
}

#[test]
fn signed_permutation_aligner_recovers_alignment() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);

    let m = 10;
    let k = 3;
    let base = random_mat(m, k, &mut rng);

    // Permutation and signs for N=4 inputs.
    let perms: [[usize; 3]; 4] = [[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1]];
    let signs_arr: [[f64; 3]; 4] = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];

    let inputs_owned: Vec<Mat<f64>> = (0..4)
        .map(|n| {
            Mat::<f64>::from_fn(m, k, |i, j| {
                signs_arr[n][j] * base[(i, perms[n][j])]
            })
        })
        .collect();
    let mat_refs: Vec<MatRef<'_, f64>> = inputs_owned.iter().map(Mat::as_ref).collect();

    let aln = generalized(
        &mat_refs,
        GpaOptions {
            inner: InnerAligner::SignedPermutation,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(aln.converged, "should have converged");
    let diff = max_pair_diff(&aln.aligned);
    assert!(diff < 1e-8, "max pairwise diff = {diff}");
}

#[test]
fn init_mean_converges_and_aligned_coincide() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(99);

    let m = 8;
    let k = 3;
    let base = random_mat(m, k, &mut rng);

    let inputs_owned: Vec<Mat<f64>> = (0..4)
        .map(|_| {
            let r = random_orthogonal(k, &mut rng);
            Mat::<f64>::from_fn(m, k, |i, j| {
                let mut v = 0.0;
                for l in 0..k {
                    v += base[(i, l)] * r[(l, j)];
                }
                v
            })
        })
        .collect();
    let mat_refs: Vec<MatRef<'_, f64>> = inputs_owned.iter().map(Mat::as_ref).collect();

    let aln = generalized(
        &mat_refs,
        GpaOptions {
            inner: InnerAligner::Orthogonal,
            init: GpaInit::Mean,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(aln.converged, "should have converged");
    let diff = max_pair_diff(&aln.aligned);
    assert!(diff < 1e-8, "max pairwise diff = {diff}");
}

#[test]
fn max_iters_cap_returns_not_converged() {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);

    let m = 8;
    let k = 3;
    let base = random_mat(m, k, &mut rng);
    let inputs_owned: Vec<Mat<f64>> = (0..3)
        .map(|_| {
            let r = random_orthogonal(k, &mut rng);
            Mat::<f64>::from_fn(m, k, |i, j| {
                let mut v = 0.0;
                for l in 0..k {
                    v += base[(i, l)] * r[(l, j)];
                }
                v
            })
        })
        .collect();
    let mat_refs: Vec<MatRef<'_, f64>> = inputs_owned.iter().map(Mat::as_ref).collect();

    let aln = generalized(
        &mat_refs,
        GpaOptions {
            max_iters: 1,
            tol: 0.0, // impossible — drift is never 0 on non-trivial input
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!aln.converged, "should not have converged with max_iters=1");
    assert_eq!(aln.n_iters, 1);
}

#[test]
fn procrustes_form_normalises_aligned() {
    // Inputs with varying magnitudes — after procrustes_form, each aligned
    // matrix should have ‖·‖_F ≈ 1.
    let scales = [1.0_f64, 5.0, 0.1, 10.0];
    let base = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 1.0 } else { 0.5 });

    let inputs_owned: Vec<Mat<f64>> = scales
        .iter()
        .map(|&s| Mat::<f64>::from_fn(4, 2, |i, j| base[(i, j)] * s))
        .collect();
    let mat_refs: Vec<MatRef<'_, f64>> = inputs_owned.iter().map(Mat::as_ref).collect();

    let aln = generalized(
        &mat_refs,
        GpaOptions {
            procrustes_form: true,
            ..Default::default()
        },
    )
    .unwrap();

    for (i, aligned) in aln.aligned.iter().enumerate() {
        let norm = frobenius(aligned);
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "aligned[{i}] ‖·‖_F = {norm}, want 1.0"
        );
    }
}

#[test]
#[allow(clippy::items_after_statements)]
fn weights_shift_consensus_toward_first_matrix() {
    // With weights [10, 1, 1] the consensus should be much closer to
    // matrices[0] than the unweighted run.
    let a0 = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { 2.0 } else { 0.0 });
    let a1 = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { -1.0 } else { 0.0 });
    let a2 = Mat::<f64>::from_fn(4, 2, |i, j| if i == j { -1.0 } else { 0.0 });

    let mats = [a0.as_ref(), a1.as_ref(), a2.as_ref()];

    let uniform = generalized(&mats, GpaOptions::default()).unwrap();
    let weighted = generalized(
        &mats,
        GpaOptions {
            weights: Some(vec![10.0, 1.0, 1.0]),
            ..Default::default()
        },
    )
    .unwrap();

    // Distance from a0 of each consensus.
    fn dist_from(consensus: &Mat<f64>, target: &Mat<f64>) -> f64 {
        let mut s = 0.0;
        for i in 0..consensus.nrows() {
            for j in 0..consensus.ncols() {
                let d = consensus[(i, j)] - target[(i, j)];
                s += d * d;
            }
        }
        s.sqrt()
    }

    let d_uniform = dist_from(&uniform.consensus, &a0);
    let d_weighted = dist_from(&weighted.consensus, &a0);

    // The heavily-weighted run must be closer to a0 than the uniform run.
    assert!(
        d_weighted < d_uniform,
        "weighted dist {d_weighted} should be < uniform dist {d_uniform}"
    );
}

// ---------------------------------------------------------------------------
// Error path tests
// ---------------------------------------------------------------------------

#[test]
fn empty_matrices_slice_returns_empty_input() {
    let empty: &[MatRef<'_, f64>] = &[];
    assert!(matches!(
        generalized(empty, GpaOptions::default()),
        Err(ProcrustesError::EmptyInput)
    ));
}

#[test]
fn zero_dimension_matrix_returns_empty_input() {
    let z = Mat::<f64>::zeros(0, 3);
    let mats = [z.as_ref()];
    assert!(matches!(
        generalized(&mats, GpaOptions::default()),
        Err(ProcrustesError::EmptyInput)
    ));
}

#[test]
fn mixed_shapes_return_dimension_mismatch() {
    let a = Mat::<f64>::zeros(4, 3);
    let b = Mat::<f64>::zeros(4, 2);
    let mats = [a.as_ref(), b.as_ref()];
    assert!(matches!(
        generalized(&mats, GpaOptions::default()),
        Err(ProcrustesError::DimensionMismatch { .. })
    ));
}

#[test]
fn weights_all_zero_returns_invalid_options() {
    let a = Mat::<f64>::zeros(3, 2);
    let mats = [a.as_ref(), a.as_ref(), a.as_ref()];
    assert!(matches!(
        generalized(
            &mats,
            GpaOptions {
                weights: Some(vec![0.0, 0.0, 0.0]),
                ..Default::default()
            }
        ),
        Err(ProcrustesError::InvalidOptions(_))
    ));
}

#[test]
fn weights_wrong_length_returns_invalid_options() {
    let a = Mat::<f64>::zeros(3, 2);
    let mats = [a.as_ref(), a.as_ref(), a.as_ref()];
    assert!(matches!(
        generalized(
            &mats,
            GpaOptions {
                weights: Some(vec![1.0, 1.0]),
                ..Default::default()
            }
        ),
        Err(ProcrustesError::InvalidOptions(_))
    ));
}

#[test]
fn weights_negative_entry_returns_invalid_options() {
    let a = Mat::<f64>::zeros(3, 2);
    let mats = [a.as_ref(), a.as_ref(), a.as_ref()];
    assert!(matches!(
        generalized(
            &mats,
            GpaOptions {
                weights: Some(vec![-1.0, 1.0, 1.0]),
                ..Default::default()
            }
        ),
        Err(ProcrustesError::InvalidOptions(_))
    ));
}

#[test]
fn weights_nan_entry_returns_invalid_options() {
    let a = Mat::<f64>::zeros(3, 2);
    let mats = [a.as_ref(), a.as_ref(), a.as_ref()];
    assert!(matches!(
        generalized(
            &mats,
            GpaOptions {
                weights: Some(vec![f64::NAN, 1.0, 1.0]),
                ..Default::default()
            }
        ),
        Err(ProcrustesError::InvalidOptions(_))
    ));
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn procrustes_form_with_zero_matrix_returns_invalid_options() {
    let a = Mat::<f64>::zeros(3, 2);
    let b = Mat::<f64>::from_fn(3, 2, |i, j| (i + j) as f64);
    let mats = [b.as_ref(), a.as_ref()];
    assert!(matches!(
        generalized(
            &mats,
            GpaOptions {
                procrustes_form: true,
                ..Default::default()
            }
        ),
        Err(ProcrustesError::InvalidOptions(_))
    ));
}
