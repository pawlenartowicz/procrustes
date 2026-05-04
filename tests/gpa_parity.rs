//! Parity vs. qc-procrustes `generalized()`
//! (<https://github.com/theochem/procrustes> — install: `pip install qc-procrustes`).

use procrustes::{generalized, GpaOptions, InnerAligner, Mat, MatRef};

/* Recipe (Python 3.10+, qc-procrustes ≥ 0.0.4):
   import numpy as np
   from procrustes import generalized
   rng = np.random.default_rng(0)
   inputs = [rng.standard_normal((6, 3)) for _ in range(4)]
   # qc-procrustes API: generalized(arrays, ref=None, ...). With ref=None
   # the consensus is iteratively refined; default tol/n_iter mirror this
   # crate's (1e-10 / 100). Confirm sign convention against a manual run
   # before pasting reference values — qc-procrustes may post-rotate the
   # consensus to align with `arrays[0]`, in which case mirror that
   # post-rotation in the assertion or document the divergence.
   result = generalized(inputs, ref=None, tol=1e-10, n_iter=100)
   print("inputs =", inputs)
   print("consensus =", result.array_new)
   print("err =", result.error)
*/

#[test]
#[ignore = "regenerate from recipe and paste reference values before un-ignoring"]
#[allow(unreachable_code)]
fn gpa_parity_qc_procrustes_4n_6m_3k() {
    // Tripwire: if someone removes `#[ignore]` before the recipe is run,
    // the placeholder zeros below would silently "pass". Fail loudly instead.
    // Delete this `panic!` (and the `#[allow(unreachable_code)]`) once real
    // fixture values are pasted.
    panic!("regenerate fixtures from recipe above before un-ignoring this test");

    // Paste from recipe output:
    let inputs: [[[f64; 3]; 6]; 4] = [[[0.0; 3]; 6]; 4];
    let expected_consensus: [[f64; 3]; 6] = [[0.0; 3]; 6];

    let mats: Vec<Mat<f64>> = inputs
        .iter()
        .map(|m| Mat::<f64>::from_fn(6, 3, |i, j| m[i][j]))
        .collect();
    let mat_refs: Vec<MatRef<'_, f64>> = mats.iter().map(Mat::as_ref).collect();

    let opts = GpaOptions {
        inner: InnerAligner::Orthogonal,
        ..Default::default()
    };
    let aln = generalized(&mat_refs, opts).unwrap();

    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        for j in 0..3 {
            assert!(
                (aln.consensus[(i, j)] - expected_consensus[i][j]).abs() < 1e-10,
                "consensus[{i},{j}] = {} expected {}",
                aln.consensus[(i, j)],
                expected_consensus[i][j]
            );
        }
    }
}
