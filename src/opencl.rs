//! Table-1 memhash acceleration using the shared MMX OpenCL implementation.
//! Both SHA-512 steps around the memory hash remain on the CPU.
use crate::{pos, Hash};
use anyhow::{ensure, Context, Result};
pub use mmx_opencl::DeviceInfo;
use rayon::prelude::*;

pub fn devices() -> Result<Vec<DeviceInfo>> {
    Ok(mmx_opencl::devices(None, true)?)
}

pub type F1Value = (u32, [u32; pos::N_META]);
pub struct MemHashOpenCl {
    engine: mmx_opencl::MemoryPipeline,
    profiles: Vec<mmx_opencl::MemoryProfile>,
}
impl MemHashOpenCl {
    pub fn new(index: usize) -> Result<Self> {
        Self::new_profiled(index, false)
    }
    /// Profiling records GPU stage timestamps; normal plotting leaves it off.
    pub fn new_profiled(index: usize, profiling: bool) -> Result<Self> {
        let mut engine =
            mmx_opencl::OpenCl::new(index, None, true)?.into_memory_pipeline(profiling)?;
        let keys = [[0; 64], [0xff; 64], std::array::from_fn(|i| i as u8)];
        engine.prepare(0, keys.len())?.copy_from_slice(&keys);
        engine.submit(0)?;
        let output = engine.finish(0)?;
        for (key, actual) in keys.iter().zip(output.hashes) {
            ensure!(
                *actual == pos::memory_hash(key),
                "OpenCL memhash self-test failed"
            );
        }
        Ok(Self {
            engine,
            profiles: Vec::new(),
        })
    }
    pub fn device(&self) -> &DeviceInfo {
        self.engine.device()
    }
    pub fn take_profiles(&mut self) -> Vec<mmx_opencl::MemoryProfile> {
        std::mem::take(&mut self.profiles)
    }

    pub fn compute_f1(
        &mut self,
        first: u32,
        count: usize,
        id: &Hash,
        ksize: u8,
    ) -> Result<Vec<F1Value>> {
        ensure!(count <= 65536, "OpenCL F1 batch exceeds 65536 entries");
        let mut result = Vec::with_capacity(count);
        self.for_each_f1(first, count as u64, id, ksize, |_, values| {
            result.extend_from_slice(values);
            Ok(())
        })?;
        Ok(result)
    }
    /// Stream ordered F1 batches through three queues without draining at host
    /// batch boundaries. The callback runs while later GPU batches are in flight.
    /// Host/device storage is bounded regardless of the total X domain length.
    pub fn for_each_f1(
        &mut self,
        first: u32,
        count: u64,
        id: &Hash,
        ksize: u8,
        mut consume: impl FnMut(u64, &[F1Value]) -> Result<()>,
    ) -> Result<()> {
        ensure!((8..=32).contains(&ksize), "ksize must be between 8 and 32");
        ensure!(
            count <= (1u64 << ksize).saturating_sub(first as u64),
            "OpenCL F1 batch exceeds X domain"
        );
        ensure!(
            first as u64 <= 1u64 << ksize,
            "OpenCL F1 batch exceeds X domain"
        );
        let result = self.run(first, count, id, ksize, &mut consume);
        if result.is_err() {
            // Callback errors include cancellation and I/O failure. Retire all
            // pending DMA before returning so this engine can safely be reused.
            self.engine
                .drain()
                .context("drain OpenCL memhash after failure")?;
        }
        result
    }
    fn run(
        &mut self,
        first: u32,
        count: u64,
        id: &Hash,
        ksize: u8,
        consume: &mut impl FnMut(u64, &[F1Value]) -> Result<()>,
    ) -> Result<()> {
        use mmx_opencl::{MEMORY_BATCH, MEMORY_PIPELINE_DEPTH};
        let batches = count.div_ceil(MEMORY_BATCH as u64) as usize;
        let mut values = vec![(0, [0; pos::N_META]); MEMORY_BATCH];
        for tick in 0..batches + MEMORY_PIPELINE_DEPTH {
            let mut retired = None;
            if tick >= MEMORY_PIPELINE_DEPTH {
                let batch = tick - MEMORY_PIPELINE_DEPTH;
                let output = self.engine.finish(batch % MEMORY_PIPELINE_DEPTH)?;
                let values = &mut values[..output.keys.len()];
                values
                    .par_iter_mut()
                    .zip(output.keys.par_iter().zip(output.hashes))
                    .with_min_len(256)
                    .try_for_each(|(dst, (key, memory))| -> Result<()> {
                        *dst = pos::f1_from_memory(key, memory, ksize)?;
                        Ok(())
                    })?;
                if let Some(profile) = output.profile {
                    self.profiles.push(profile);
                }
                retired = Some((batch, values.len()));
            }
            if tick < batches {
                let offset = tick as u64 * MEMORY_BATCH as u64;
                let len = (count - offset).min(MEMORY_BATCH as u64) as usize;
                let keys = self.engine.prepare(tick % MEMORY_PIPELINE_DEPTH, len)?;
                keys.par_iter_mut()
                    .enumerate()
                    .with_min_len(256)
                    .for_each(|(i, key)| {
                        *key = pos::f1_key((first as u64 + offset + i as u64) as u32, id)
                    });
                self.engine.submit(tick % MEMORY_PIPELINE_DEPTH)?;
            }
            if let Some((batch, len)) = retired {
                // Refill the vacated stream before calling the sorter, so all
                // three queues can advance during a slow callback or disk write.
                consume(
                    first as u64 + batch as u64 * MEMORY_BATCH as u64,
                    &values[..len],
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_device_matches_cpu_at_batch_and_production_domain_boundaries() {
        if std::env::var("MMX_OPENCL_TESTS").as_deref() != Ok("1") {
            return;
        }
        let mut gpu = MemHashOpenCl::new(0).expect("OpenCL GPU required by MMX_OPENCL_TESTS=1");
        let id = std::array::from_fn(|i| (i * 137) as u8);
        // Cross the shared backend's 4096-key scratch batch, and exercise reuse.
        for (k, first, count) in [
            (16, 23, 4097),
            (16, 31, 3 * mmx_opencl::MEMORY_BATCH + 17),
            (17, 7, 65536),
            (8, 249, 7),
            (29, (1 << 29) - 7, 7),
            (32, u32::MAX - 6, 7),
            (16, 1, 1),
            (16, 0, 0),
        ] {
            let expected: Vec<_> = (0..count)
                .into_par_iter()
                .map(|i| pos::compute_f1(first + i as u32, &id, k).unwrap())
                .collect();
            assert_eq!(gpu.compute_f1(first, count, &id, k).unwrap(), expected);
        }
        assert!(gpu.compute_f1(0, 65537, &id, 29).is_err());
        assert!(gpu.compute_f1(u32::MAX, 2, &id, 32).is_err());
        assert!(gpu.compute_f1(0, 1, &id, 7).is_err());
    }
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;
    #[test]
    fn callback_failure_drains_inflight_batches_and_allows_reuse() {
        if std::env::var("MMX_OPENCL_TESTS").as_deref() != Ok("1") {
            return;
        }
        let mut gpu = MemHashOpenCl::new(0).unwrap();
        let mut calls = 0;
        let error = gpu
            .for_each_f1(0, 65537, &[3; 32], 17, |_, _| {
                calls += 1;
                anyhow::bail!("cancelled by test")
            })
            .unwrap_err();
        assert!(error.to_string().contains("cancelled by test"));
        assert_eq!(calls, 1);
        let actual = gpu.compute_f1(17, 4097, &[7; 32], 29).unwrap();
        let expected: Vec<_> = (17..17 + 4097)
            .into_par_iter()
            .map(|x| pos::compute_f1(x, &[7; 32], 29).unwrap())
            .collect();
        assert_eq!(actual, expected);
    }
    #[test]
    fn streaming_crosses_old_host_batch_boundary_in_order() {
        if std::env::var("MMX_OPENCL_TESTS").as_deref() != Ok("1") {
            return;
        }
        let mut gpu = MemHashOpenCl::new_profiled(0, true).unwrap();
        let mut actual = Vec::new();
        gpu.for_each_f1(23, 65537, &[9; 32], 17, |first, values| {
            assert_eq!(first, 23 + actual.len() as u64);
            actual.extend_from_slice(values);
            Ok(())
        })
        .unwrap();
        let expected: Vec<_> = (23..23 + 65537)
            .into_par_iter()
            .map(|x| pos::compute_f1(x, &[9; 32], 17).unwrap())
            .collect();
        assert_eq!(actual, expected);
        let profiles = gpu.take_profiles();
        assert_eq!(
            profiles.len(),
            65537usize.div_ceil(mmx_opencl::MEMORY_BATCH)
        );
        for (i, profile) in profiles.iter().enumerate() {
            assert_eq!(profile.slot, i % mmx_opencl::MEMORY_PIPELINE_DEPTH);
            for stage in &profile.stages {
                assert!(stage.start <= stage.end);
            }
            for stages in profile.stages.windows(2) {
                assert!(stages[0].end <= stages[1].start);
            }
        }
    }
}
