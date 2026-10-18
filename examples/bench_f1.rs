//! Isolate leaf hashing from sorting and file I/O for CPU tuning.
use mmx_cpu_plotter::pos;
use rayon::prelude::*;
use std::{hint::black_box, time::Instant};

fn main() {
    let mut args = std::env::args().skip(1);
    let count: usize = args.next().map_or(200_000, |s| s.parse().unwrap());
    let threads: usize = args.next().map_or(1, |s| s.parse().unwrap());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let start = Instant::now();
    let checksum = pool.install(|| {
        (0..count)
            .into_par_iter()
            .with_min_len(256)
            .map(|x| {
                let (y, meta) = black_box(pos::compute_f1(x as u32, &[0; 32], 29).unwrap());
                (y as u64) << 32 | meta[0] as u64
            })
            .reduce(|| 0, |a, b| a ^ b)
    });
    let seconds = start.elapsed().as_secs_f64();
    println!("count={count} threads={threads} seconds={seconds:.6} x_per_second={:.0} checksum={checksum:016x}",count as f64/seconds);
}
