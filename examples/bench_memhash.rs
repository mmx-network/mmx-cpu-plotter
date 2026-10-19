//! Compare complete F1 work with CPU-only and OpenCL memhash, including transfers.
use anyhow::{ensure, Result};
use mmx_cpu_plotter::{opencl::MemHashOpenCl, pos};
use rayon::prelude::*;
use std::time::Instant;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let count: usize = args.first().map_or(Ok(200000), |s| s.parse())?;
    let threads: usize = args.get(1).map_or(Ok(4), |s| s.parse())?;
    let device: usize = args.get(2).map_or(Ok(0), |s| s.parse())?;
    ensure!(
        count > 0 && count <= 1 << 29 && threads > 0,
        "invalid count or threads"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut gpu = MemHashOpenCl::new(device)?;
    println!(
        "GPU {}: {}; {count} F1 values at k29; {threads} CPU threads",
        device,
        gpu.device().name
    );
    let id = [0; 32];
    let expected: Vec<_> = pool.install(|| {
        (0..count)
            .into_par_iter()
            .with_min_len(256)
            .map(|x| pos::compute_f1(x as u32, &id, 29).unwrap())
            .collect()
    });
    // Warm full-size device scratch and host work independently of timed rounds.
    pool.install(|| gpu.compute_f1(0, count.min(65536), &id, 29))?;
    for round in 1..=3 {
        let start = Instant::now();
        let actual: Vec<_> = pool.install(|| {
            (0..count)
                .into_par_iter()
                .with_min_len(256)
                .map(|x| pos::compute_f1(x as u32, &id, 29).unwrap())
                .collect()
        });
        let cpu_time = start.elapsed().as_secs_f64();
        ensure!(actual == expected, "CPU F1 mismatch");
        let start = Instant::now();
        let actual = pool.install(|| -> Result<Vec<_>> {
            let mut output = Vec::with_capacity(count);
            for first in (0..count).step_by(65536) {
                output.extend(gpu.compute_f1(first as u32, (count - first).min(65536), &id, 29)?);
            }
            Ok(output)
        })?;
        let gpu_time = start.elapsed().as_secs_f64();
        ensure!(actual == expected, "OpenCL F1 mismatch");
        println!("round {round}: CPU {cpu_time:.4}s ({:.0} X/s), GPU memhash {gpu_time:.4}s ({:.0} X/s), {:.2}x", count as f64 / cpu_time, count as f64 / gpu_time, cpu_time / gpu_time);
    }
    Ok(())
}
