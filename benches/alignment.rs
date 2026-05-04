//! Wall-time benches for the four public alignment entry points.
//!
//! Sweep covers `K ∈ {2, 3, 4, 6, 10, 16}` so the `signed_permutation` bench
//! exercises both the brute-force path (`K ≤ BRUTE_FORCE_CUTOFF`, currently 3)
//! and the Jonker-Volgenant path (`K > BRUTE_FORCE_CUTOFF`). `M = 32` matches
//! the bit-parity test fixtures.
//!
//! Run all benches:
//! ```text
//! cargo bench --bench alignment
//! ```
//!
//! Run a single cell, fast (≈2 s warm-up + measurement):
//! ```text
//! cargo bench --bench alignment -- \
//!     --warm-up-time 1 --measurement-time 1 'orthogonal/M=32/K=4'
//! ```

// criterion_group! generates functions that cannot carry doc-comments.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};
use procrustes::Mat;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const M: usize = 32;
const K_SWEEP: &[usize] = &[2, 3, 4, 6, 10, 16];
const SEED: u64 = 0xC0_FF_EE;

fn synth_pair(m: usize, k: usize, seed: u64) -> (Mat<f64>, Mat<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let a = Mat::<f64>::from_fn(m, k, |_, _| rng.gen_range(-1.0..1.0));
    let reference = Mat::<f64>::from_fn(m, k, |_, _| rng.gen_range(-1.0..1.0));
    (a, reference)
}

fn bench_orthogonal(c: &mut Criterion) {
    for &k in K_SWEEP {
        let (a, reference) = synth_pair(M, k, SEED);
        c.bench_function(&format!("orthogonal/M={M}/K={k}"), |bencher| {
            bencher.iter(|| procrustes::orthogonal(a.as_ref(), reference.as_ref(), false).unwrap());
        });
    }
}

fn bench_rotation_only(c: &mut Criterion) {
    for &k in K_SWEEP {
        let (a, reference) = synth_pair(M, k, SEED);
        c.bench_function(&format!("rotation_only/M={M}/K={k}"), |bencher| {
            bencher
                .iter(|| procrustes::rotation_only(a.as_ref(), reference.as_ref(), false).unwrap());
        });
    }
}

fn bench_signed_permutation(c: &mut Criterion) {
    for &k in K_SWEEP {
        let (a, reference) = synth_pair(M, k, SEED);
        c.bench_function(&format!("signed_permutation/M={M}/K={k}"), |bencher| {
            bencher.iter(|| {
                procrustes::signed_permutation(a.as_ref(), reference.as_ref(), false).unwrap()
            });
        });
    }
}

fn bench_sign_align(c: &mut Criterion) {
    for &k in K_SWEEP {
        let (a, reference) = synth_pair(M, k, SEED);
        c.bench_function(&format!("sign_align/M={M}/K={k}"), |bencher| {
            bencher.iter(|| procrustes::sign_align(a.as_ref(), reference.as_ref(), false).unwrap());
        });
    }
}

criterion_group!(
    benches,
    bench_orthogonal,
    bench_rotation_only,
    bench_signed_permutation,
    bench_sign_align,
);
criterion_main!(benches);
