//! Bit-parity tripwire — guards against faer-side numerical drift.
//!
//! These tests pin the exact `f64` bit pattern of every continuous output for
//! a deterministic input pair, captured under `faer = 0.24.0` on
//! `x86_64-unknown-linux-gnu`. A failure on a Dependabot PR that bumps faer
//! is the signal that faer's SVD / arithmetic changed even sub-ULP, and means
//! procrustes needs a major version bump (numerical contract changed) — even
//! when the approximate-tolerance tests in `scipy_parity.rs` still pass.
//!
//! Gated to `x86_64` because IEEE 754 ops are bit-exact across `x86_64` hosts
//! but can drift on other architectures (ARM FMA pairing, BLAS backends).
//!
//! Regeneration recipe: temporarily replace each `assert_eq!(x.to_bits(), …)`
//! with `eprintln!("{:#018x}", x.to_bits());`, run
//! `cargo test --test bit_parity -- --nocapture`, and paste the printed
//! `0x…u64` literals back into the constants below.

#![cfg(target_arch = "x86_64")]

use procrustes::{orthogonal, signed_permutation, Mat};

#[test]
#[allow(clippy::needless_range_loop)]
fn bit_parity_orthogonal() {
    let a = Mat::<f64>::from_fn(4, 3, |i, j| {
        let rows = [
            [1.0, 2.0, -1.0],
            [0.5, -0.5, 1.5],
            [-1.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ];
        rows[i][j]
    });
    let reference = Mat::<f64>::from_fn(4, 3, |i, j| {
        let rows = [
            [1.5, 0.7, -0.3],
            [0.1, 0.6, 1.0],
            [-0.8, 1.2, 0.4],
            [1.9, -0.2, 1.1],
        ];
        rows[i][j]
    });

    let aln = orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();

    // Captured under faer 0.24.0, x86_64-linux.
    let expected_r_bits: [[u64; 3]; 3] = [
        [
            0x3fed_d80e_dab6_55b2,
            0xbfd3_a104_e493_7004,
            0x3fc8_55d4_3b24_92e3,
        ],
        [
            0x3fd5_dbfb_fc35_01ed,
            0x3fed_72eb_70d8_3a41,
            0xbfc8_6e55_a405_973d,
        ],
        [
            0xbfbd_cde8_9704_8b5b,
            0x3fcf_18c0_dc1d_5fb5,
            0x3fee_d121_af57_598f,
        ],
    ];
    let expected_scale_bits: u64 = 0x4028_06c6_8d18_029f;
    let expected_residual_bits: u64 = 0x3ffb_d240_55c6_9dfa;

    for i in 0..3 {
        for j in 0..3 {
            let got = aln.rotation[(i, j)].to_bits();
            assert_eq!(
                got, expected_r_bits[i][j],
                "R[{i},{j}] bits drift: got {got:#018x}, want {:#018x} \
                 (faer numerics changed — bump procrustes major)",
                expected_r_bits[i][j]
            );
        }
    }
    assert_eq!(
        aln.scale.to_bits(),
        expected_scale_bits,
        "scale bits drift (faer numerics changed)"
    );

    let resid = aln.residual_frobenius(a.as_ref(), reference.as_ref());
    assert_eq!(
        resid.to_bits(),
        expected_residual_bits,
        "residual_frobenius bits drift (faer numerics changed)"
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn bit_parity_signed_permutation() {
    let a = Mat::<f64>::from_fn(5, 3, |i, j| {
        let rows = [
            [0.5, 0.7, -0.3],
            [-0.1, 0.6, 1.0],
            [-0.8, 1.2, 0.4],
            [1.9, -0.2, 1.1],
            [0.3, 0.4, -0.5],
        ];
        rows[i][j]
    });
    // reference is a known signed permutation of a's columns plus mild drift,
    // forcing the algorithm to choose the right (perm, signs) pair under noise.
    let reference = Mat::<f64>::from_fn(5, 3, |i, j| match j {
        0 => -a[(i, 1)] + 0.01 * (i as f64),
        1 => a[(i, 2)] - 0.02 * (i as f64),
        2 => a[(i, 0)] + 0.005 * (i as f64),
        _ => unreachable!(),
    });

    let aln = signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap();

    // Discrete outputs — exact equality, no bits-conversion needed.
    assert_eq!(aln.assigned, vec![1_usize, 2, 0]);
    assert_eq!(aln.signs, vec![-1.0_f64, 1.0, 1.0]);

    // Continuous output — pinned bit-exact.
    let expected_residual_bits: u64 = 0x3fc0_1059_f2e3_38c4;
    assert_eq!(
        aln.residual_frobenius.to_bits(),
        expected_residual_bits,
        "residual_frobenius bits drift (faer or arithmetic changed)"
    );
}

#[test]
#[allow(clippy::cast_precision_loss)]
#[allow(clippy::float_cmp)] // intentional bit-exact equality check for all-distinct sanity
fn bit_parity_signed_permutation_jv_path() {
    // K=10 forces the JV path (K > BRUTE_FORCE_CUTOFF=3).
    // Construct a deterministic input with a unique optimum: integer-valued
    // `a` and `reference` chosen so |dot[i, j]| values are all distinct,
    // eliminating tie-breaking ambiguity between paths.

    // Construction: 24×10 integer-valued `a` from a fixed RNG seed; reference
    // is built by applying a known signed permutation of a's columns plus
    // small integer drift, ensuring a unique winning (perm, signs) pair.

    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xBEEF_CAFE_F00D_1234);
    let m = 24;
    let k = 10;

    // Integer entries in [-9, 9] so |dot[i, j]| values land at integers from
    // sums of integer products — distinct |dot| values are likely; we add a
    // small position-dependent drift below to break any residual ties.
    let a = Mat::<f64>::from_fn(m, k, |_, _| f64::from(rng.gen_range(-9_i32..=9)));

    let true_perm: Vec<usize> = vec![3, 7, 0, 9, 1, 5, 8, 2, 6, 4];
    let true_signs: Vec<f64> = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

    let reference = Mat::<f64>::from_fn(m, k, |i, j| {
        true_signs[j] * a[(i, true_perm[j])] + 0.001 * ((i as f64) + 0.5 * (j as f64))
    });

    // Sanity: ensure all |dot[i, j]| values are distinct (tie-elimination check).
    {
        let mut dot_abs = Vec::with_capacity(k * k);
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0;
                for r in 0..m {
                    s += a[(r, i)] * reference[(r, j)];
                }
                dot_abs.push(s.abs());
            }
        }
        let mut sorted = dot_abs.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut all_distinct = true;
        for w in sorted.windows(2) {
            if w[0] == w[1] {
                all_distinct = false;
                break;
            }
        }
        assert!(
            all_distinct,
            "construction failed to produce all-distinct |dot| values — adjust seed/integers/drift"
        );
    }

    let aln = signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap();

    // Pinned outputs captured on first run (x86_64-linux, faer 0.24.0, JV path K=10).
    assert_eq!(aln.assigned, vec![3_usize, 7, 0, 9, 1, 5, 8, 2, 6, 4]);
    assert_eq!(
        aln.signs,
        vec![1.0_f64, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    );
    let expected_residual_bits: u64 = 0x3fce_a89a_5bab_157c;
    assert_eq!(
        aln.residual_frobenius.to_bits(),
        expected_residual_bits,
        "residual_frobenius bits drift (JV path arithmetic or faer changed)"
    );
}
