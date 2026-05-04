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

/* Regeneration recipe (Python ≥ 3.10, scipy ≥ 1.10):

   import numpy as np
   from scipy.linalg import orthogonal_procrustes
   a = np.array([[(i - 3.5) * 0.5] for i in range(8)], dtype=float)
   b = np.array([[((i * 3) % 7) - 3.0] for i in range(8)], dtype=float)
   R, scale = orthogonal_procrustes(a, b)
   np.set_printoptions(precision=17, floatmode='unique')
   print("R =", R.tolist())
   print("scale =", scale)
*/
#[allow(clippy::cast_precision_loss)]
#[test]
fn scipy_parity_k_eq_1() {
    let a = Mat::<f64>::from_fn(8, 1, |i, _| ((i as f64) - 3.5) * 0.5);
    let reference = Mat::<f64>::from_fn(8, 1, |i, _| (((i * 3) % 7) as f64) - 3.0);

    // SciPy reference values (scipy 1.16.3, generated 2026-05-01).
    let expected_r: f64 = -1.0;
    let expected_scale: f64 = 1.75;

    let aln = procrustes::orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();

    assert!(
        (aln.rotation[(0, 0)] - expected_r).abs() < 1e-10,
        "R[0,0] = {} want {}",
        aln.rotation[(0, 0)],
        expected_r
    );
    assert!(
        (aln.scale - expected_scale).abs() < 1e-10,
        "scale = {} want {}",
        aln.scale,
        expected_scale
    );
}

/* Regeneration recipe (Python ≥ 3.10, scipy ≥ 1.10):

   import numpy as np
   from scipy.linalg import orthogonal_procrustes
   M = np.array([[float(((i*4 + j*7 + 11) % 13) - 6) for j in range(4)] for i in range(6)])
   Qa, _ = np.linalg.qr(M)
   sigma = np.array([1.0, 1.0, 1.0, 0.001])
   a = Qa
   b = Qa @ np.diag(sigma)
   R, scale = orthogonal_procrustes(a, b)
   np.set_printoptions(precision=17, floatmode='unique')
   print("a =", a.tolist())
   print("R =", R.tolist())
   print("scale =", scale)
   # Sanity: singular values of a^T b
   print("svd(a^T b) =", np.linalg.svd(a.T @ b, compute_uv=False).tolist())
*/
#[test]
fn scipy_parity_near_degenerate() {
    // a: 6×4 with orthonormal columns (output of numpy.linalg.qr on a
    // deterministic 6×4 integer matrix — values hardcoded since faer's QR
    // need not produce the same signs/order).
    #[rustfmt::skip]
    let a_vals: [[f64; 4]; 6] = [
        [-0.548_821_299_948_451_6, -0.176_002_570_100_906_75, -0.421_452_485_589_294_2, -0.315_264_144_377_731_5],
        [ 0.439_057_039_958_761_4, -0.122_624_741_463_746_47, -0.380_963_600_510_846_9, -0.405_339_614_199_940_5],
        [ 0.0,                      0.718_436_720_575_832_5, -0.462_861_572_601_343,    0.472_896_216_566_597_2],
        [-0.439_057_039_958_761_4,  0.002_885_288_034_441_116, -0.429_734_302_991_704_1, -0.157_632_072_188_865_77],
        [ 0.548_821_299_948_451_7,  0.056_263_116_671_601_3, -0.389_245_417_913_256_6, -0.247_707_542_011_074_76],
        [ 0.109_764_259_989_690_35, -0.659_288_315_869_79,    -0.356_118_148_303_617_77, 0.653_047_156_211_015_2],
    ];
    let a = Mat::<f64>::from_fn(6, 4, |i, j| a_vals[i][j]);

    // reference = a · diag(1, 1, 1, 0.001)
    let sigma = [1.0, 1.0, 1.0, 0.001];
    let reference = Mat::<f64>::from_fn(6, 4, |i, j| a_vals[i][j] * sigma[j]);

    // SciPy reference values (scipy 1.16.3, generated 2026-05-01). Because
    // a^T · b = diag(sigma), its SVD has U = V = I (modulo sign/order under
    // clustered SVs), so R is identity to within 1e-15. The test guards
    // against any future SVD path that returns large numerical noise on
    // clustered singular values.
    let expected_r: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let expected_scale: f64 = 3.001; // = 1 + 1 + 1 + 0.001

    let aln = procrustes::orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();

    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        for j in 0..4 {
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

/* Regeneration recipe (Python ≥ 3.10, scipy ≥ 1.10):

   import numpy as np
   from scipy.linalg import orthogonal_procrustes
   M, K = 100, 2
   a = np.zeros((M, K))
   b = np.zeros((M, K))
   for i in range(M):
       for j in range(K):
           ax = ((i + 1) * (j + 2)) % 17 - 8
           ay = ((i * 3 + j * 5) % 11) - 5
           a[i, j] = ax * 0.1 + ay * 0.05
           bx = ((i + 3) * (j + 1)) % 19 - 9
           by = ((i * 7 + j * 11) % 13) - 6
           b[i, j] = bx * 0.07 + by * 0.04
   R, scale = orthogonal_procrustes(a, b)
   np.set_printoptions(precision=17, floatmode='unique')
   print("R =", R.tolist())
   print("scale =", scale)
*/
#[allow(clippy::cast_precision_loss)]
#[test]
fn scipy_parity_tall_skinny() {
    let m: usize = 100;
    let k: usize = 2;
    let a = Mat::<f64>::from_fn(m, k, |i, j| {
        let ax = (((i + 1) * (j + 2)) % 17) as f64 - 8.0;
        let ay = ((i * 3 + j * 5) % 11) as f64 - 5.0;
        ax * 0.1 + ay * 0.05
    });
    let reference = Mat::<f64>::from_fn(m, k, |i, j| {
        let bx = (((i + 3) * (j + 1)) % 19) as f64 - 9.0;
        let by = ((i * 7 + j * 11) % 13) as f64 - 6.0;
        bx * 0.07 + by * 0.04
    });

    // SciPy reference values (scipy 1.16.3, generated 2026-05-01).
    let expected_r: [[f64; 2]; 2] = [
        [-0.269_012_263_893_954_05, 0.963_136_751_388_217_7],
        [0.963_136_751_388_217_9, 0.269_012_263_893_954_1],
    ];
    let expected_scale: f64 = 0.960_922_733_626_384_1;

    let aln = procrustes::orthogonal(a.as_ref(), reference.as_ref(), false).unwrap();

    #[allow(clippy::needless_range_loop)]
    for i in 0..2 {
        for j in 0..2 {
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
