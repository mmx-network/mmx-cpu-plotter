//! Compare complete F1 work with CPU-only and OpenCL memhash, including transfers.
use anyhow::{ensure, Result};
use mmx_cpu_plotter::{
    opencl::{F1Value, MemHashOpenCl},
    pos,
};
use rayon::prelude::*;
use sha2::{Digest, Sha512};
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
    let mut blocking = mmx_opencl::OpenCl::new(device, None, true)?;
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
    pool.install(|| blocking_f1(&mut blocking, count.min(65536), &id))?;
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
        let actual = pool.install(|| blocking_f1(&mut blocking, count, &id))?;
        let blocking_time = start.elapsed().as_secs_f64();
        ensure!(actual == expected, "blocking OpenCL F1 mismatch");
        let start = Instant::now();
        let actual = pool.install(|| -> Result<Vec<_>> {
            let mut output = Vec::with_capacity(count);
            gpu.for_each_f1(0, count as u64, &id, 29, |_, values| {
                output.extend_from_slice(values);
                Ok(())
            })?;
            Ok(output)
        })?;
        let gpu_time = start.elapsed().as_secs_f64();
        ensure!(actual == expected, "OpenCL F1 mismatch");
        println!("round {round}: CPU {cpu_time:.4}s ({:.0} X/s), blocking {blocking_time:.4}s ({:.0} X/s), pipeline {gpu_time:.4}s ({:.0} X/s), {:.2}x vs CPU, {:.2}x vs blocking", count as f64 / cpu_time, count as f64 / blocking_time, count as f64 / gpu_time, cpu_time / gpu_time, blocking_time / gpu_time);
    }
    // Profile outside the timed runs. Record actual device overlap, including
    // transfers versus kernels, rather than inferring it from host submissions.
    let mut profiled = MemHashOpenCl::new_profiled(device, true)?;
    pool.install(|| profiled.for_each_f1(0, count.min(65536) as u64, &id, 29, |_, _| Ok(())))?;
    let profiles = profiled.take_profiles();
    let mut kernel_overlap = 0;
    let mut copy_overlap = 0;
    for (i, a) in profiles.iter().enumerate() {
        for b in &profiles[i + 1..] {
            if a.slot == b.slot {
                continue;
            }
            for (ai, x) in a.stages.iter().enumerate() {
                for (bi, y) in b.stages.iter().enumerate() {
                    let ns = x.end.min(y.end).saturating_sub(x.start.max(y.start));
                    if (ai == 1 || ai == 2) && (bi == 1 || bi == 2) {
                        kernel_overlap += ns;
                    }
                    if (ai == 0 || ai == 3) != (bi == 0 || bi == 3) {
                        copy_overlap += ns;
                    }
                }
            }
        }
    }
    println!("device profiling: {} batches, cross-queue kernel overlap {:.3} ms, copy/compute overlap {:.3} ms", profiles.len(), kernel_overlap as f64 / 1e6, copy_overlap as f64 / 1e6);

    Ok(())
}

// Former synchronous execution order, retained here only as a timing baseline.
fn blocking_f1(
    engine: &mut mmx_opencl::OpenCl,
    count: usize,
    id: &[u8; 32],
) -> Result<Vec<F1Value>> {
    let mut output = Vec::with_capacity(count);
    for first in (0..count).step_by(65536) {
        let n = (count - first).min(65536);
        let keys: Vec<[u8; 64]> = (first..first + n)
            .into_par_iter()
            .with_min_len(256)
            .map(|x| {
                let mut message = [0; 36];
                message[..4].copy_from_slice(&(x as u32).to_le_bytes());
                message[4..].copy_from_slice(id);
                Sha512::digest(message).into()
            })
            .collect();
        let memory = engine.memory_hashes(&keys)?;
        let values: Vec<_> = keys
            .par_iter()
            .zip(&memory)
            .with_min_len(256)
            .map(|(key, memory)| {
                let mut message = [0; 192];
                message[..64].copy_from_slice(key);
                message[64..].copy_from_slice(memory);
                let hash = Sha512::digest(message);
                let mut y = 0;
                let meta = std::array::from_fn(|i| {
                    let word = u32::from_le_bytes(hash[i * 4..i * 4 + 4].try_into().unwrap());
                    y ^= word;
                    word & ((1 << 29) - 1)
                });
                (y & ((1 << 29) - 1), meta)
            })
            .collect();
        output.extend(values);
    }
    Ok(output)
}
