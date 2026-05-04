//! Demonstrate `procrustes::orthogonal` and `procrustes::signed_permutation`
//! on random input where ground truth is known.
//!
//! Run with: `cargo run --example align_random --release`

use procrustes::Mat;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x_A11_60D);
    let m = 32_usize;
    let k = 4_usize;

    let reference = Mat::<f64>::from_fn(m, k, |_, _| rng.gen_range(-1.0..1.0));

    // ---- orthogonal: apply a known rotation, recover it. -------------------
    // R0 = U·Vᵀ from the SVD of a random K×K matrix is orthogonal.
    let g = Mat::<f64>::from_fn(k, k, |_, _| rng.gen_range(-1.0..1.0));
    let svd = g.as_ref().svd().expect("svd of random matrix");
    let mut r0 = Mat::<f64>::zeros(k, k);
    faer::linalg::matmul::matmul(
        r0.as_mut(),
        faer::Accum::Replace,
        svd.U(),
        svd.V().transpose(),
        1.0,
        faer::Par::Seq,
    );
    let a_rot: Mat<f64> = &reference * &r0;

    let aln = procrustes::orthogonal(a_rot.as_ref(), reference.as_ref(), true)
        .expect("orthogonal Procrustes");
    println!("orthogonal: scale = {:.6}", aln.scale);
    println!(
        "orthogonal: residual_F = {:.3e}",
        aln.residual_frobenius(a_rot.as_ref(), reference.as_ref())
    );

    // ---- signed_permutation: apply a known signed permutation, recover. ----
    let true_perm = [3_usize, 0, 2, 1];
    let true_signs = [1.0_f64, -1.0, 1.0, -1.0];
    let a_perm = Mat::<f64>::from_fn(m, k, |i, j| true_signs[j] * reference[(i, true_perm[j])]);

    let sp = procrustes::signed_permutation(a_perm.as_ref(), reference.as_ref(), true)
        .expect("signed permutation");
    println!("signed_permutation: assigned = {:?}", sp.assigned);
    println!("signed_permutation: signs    = {:?}", sp.signs);
    println!(
        "signed_permutation: residual_F = {:.3e}",
        sp.residual_frobenius
    );

    assert!(sp.residual_frobenius < 1e-10);
}
