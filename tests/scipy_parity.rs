//! `SciPy` parity for `orthogonal`. Reference values precomputed via
//! `scipy.linalg.orthogonal_procrustes` (see commented snippet below) and
//! hardcoded; no live `SciPy` invocation.

/* Regeneration recipe (Python ≥ 3.10, scipy ≥ 1.10):

   import numpy as np
   from scipy.linalg import orthogonal_procrustes

   a1 = np.array([
       [ 1.0,  2.0, -1.0],
       [ 0.5, -0.5,  1.5],
       [-1.0,  1.0,  0.0],
       [ 2.0,  0.0,  1.0],
   ])
   b1 = np.array([
       [ 1.5,  0.7, -0.3],
       [ 0.1,  0.6,  1.0],
       [-0.8,  1.2,  0.4],
       [ 1.9, -0.2,  1.1],
   ])
   R1, scale1 = orthogonal_procrustes(a1, b1)
   np.set_printoptions(precision=17, floatmode='unique')
   print("R1 =", R1.tolist())
   print("scale1 =", scale1)
*/

use procrustes::Mat;

#[test]
fn scipy_parity_fixture_1() {
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

    // SciPy reference values (scipy 1.16.3, generated 2026-04-30).
    let expected_r: [[f64; 3]; 3] = [
        [
            0.932_624_270_603_516_6,
            -0.306_702_826_708_488_14,
            0.190_119_294_050_552_7,
        ],
        [
            0.341_551_777_168_233_43,
            0.920_278_282_546_697_2,
            -0.190_867_142_761_982_63,
        ],
        [
            -0.116_423_165_196_960_72,
            0.242_942_912_557_508_05,
            0.963_028_757_537_175_5,
        ],
    ];
    let expected_scale: f64 = 12.013_233_575_039_8;

    let aln = procrustes::orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();

    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (aln.rotation[(i, j)] - expected_r[i][j]).abs() < 1e-10,
                "R[{i},{j}] = {} want {}",
                aln.rotation[(i, j)],
                expected_r[i][j]
            );
        }
    }
    assert!(
        (aln.scale - expected_scale).abs() < 1e-10,
        "scale = {} want {}",
        aln.scale,
        expected_scale
    );
}
