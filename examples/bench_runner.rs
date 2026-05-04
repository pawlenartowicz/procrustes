//! Subprocess bench worker — read two column-major `f64` matrices from disk,
//! run a procrustes op `reps` times, print one JSON line with timing + residual.
//!
//! Invoked by `../../benchmarks/bench_procrustes.py`. Not part of the public
//! API; it exists so the comparison harness can run procrustes on the same
//! seeded data as scipy/numpy without re-implementing matrix I/O.
//!
//! Wire format for each input file (little-endian):
//!     [u64 nrows][u64 ncols][f64 * nrows * ncols]   // column-major
//!
//! Run manually:
//!     cargo run --release --example `bench_runner` -- \
//!         orthogonal /tmp/a.bin /tmp/b.bin 1000

use std::env;
use std::fs::File;
use std::io::Read;
use std::time::Instant;

use procrustes::{Mat, MatRef};

// Bench worker runs on the host (amd64); 32-bit platforms are not a concern.
#[allow(clippy::cast_possible_truncation)]
fn read_mat(path: &str) -> Mat<f64> {
    let mut f = File::open(path).expect("open input");
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).expect("read header");
    let nrows = u64::from_le_bytes(hdr[..8].try_into().unwrap()) as usize;
    let ncols = u64::from_le_bytes(hdr[8..].try_into().unwrap()) as usize;
    let mut buf = vec![0u8; nrows * ncols * 8];
    f.read_exact(&mut buf).expect("read data");
    let data: Vec<f64> = buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    MatRef::from_column_major_slice(&data, nrows, ncols).to_owned()
}

fn run_once(op: &str, a: MatRef<'_, f64>, b: MatRef<'_, f64>) {
    match op {
        "orthogonal" => {
            procrustes::orthogonal(a, b, false).expect("orthogonal");
        }
        "rotation_only" => {
            procrustes::rotation_only(a, b, false).expect("rotation_only");
        }
        "signed_permutation" => {
            procrustes::signed_permutation(a, b, false).expect("signed_permutation");
        }
        "sign_align" => {
            procrustes::sign_align(a, b, false).expect("sign_align");
        }
        _ => panic!("unknown op {op}"),
    }
}

fn residual_for(op: &str, a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    match op {
        "orthogonal" => procrustes::orthogonal(a, b, false)
            .unwrap()
            .residual_frobenius(a, b),
        "rotation_only" => procrustes::rotation_only(a, b, false)
            .unwrap()
            .residual_frobenius(a, b),
        "signed_permutation" => {
            procrustes::signed_permutation(a, b, false)
                .unwrap()
                .residual_frobenius
        }
        "sign_align" => {
            procrustes::sign_align(a, b, false)
                .unwrap()
                .residual_frobenius
        }
        _ => f64::NAN,
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

// `as_nanos() as f64`: precision loss only above ~52 days of elapsed time, n/a here.
#[allow(clippy::cast_precision_loss)]
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: bench_runner <op> <a.bin> <b.bin> <reps>");
        std::process::exit(2);
    }
    let op = args[1].as_str();
    let a = read_mat(&args[2]);
    let b = read_mat(&args[3]);
    let reps: usize = args[4].parse().expect("reps");

    run_once(op, a.as_ref(), b.as_ref()); // warm-up (excluded)

    let mut times_us = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        run_once(op, a.as_ref(), b.as_ref());
        times_us.push(t0.elapsed().as_nanos() as f64 / 1000.0);
    }
    let residual = residual_for(op, a.as_ref(), b.as_ref());
    let med = median(times_us.clone());
    let min = times_us.iter().copied().fold(f64::INFINITY, f64::min);

    println!(
        "{{\"backend\":\"procrustes-rs\",\"op\":\"{op}\",\
         \"m\":{m},\"k\":{k},\"reps\":{reps},\
         \"median_us\":{med:.6},\"min_us\":{min:.6},\
         \"residual\":{residual:.12e}}}",
        m = a.nrows(),
        k = a.ncols(),
    );
}
