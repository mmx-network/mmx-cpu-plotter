//! Table-1 memhash acceleration using the shared MMX OpenCL implementation.
//! Both SHA-512 steps around the memory hash remain on the CPU.
use crate::{pos, Hash};
use anyhow::{ensure, Context, Result};
pub use mmx_opencl::DeviceInfo;
use rayon::prelude::*;

pub fn devices() -> Result<Vec<DeviceInfo>> {
    Ok(mmx_opencl::devices(None, true)?)
}

pub struct MemHashOpenCl {
    engine: mmx_opencl::OpenCl,
}
impl MemHashOpenCl {
    pub fn new(index: usize) -> Result<Self> {
        let mut engine = mmx_opencl::OpenCl::new(index, None, true)?;
        let keys = [[0; 64], [0xff; 64], std::array::from_fn(|i| i as u8)];
        for (key, actual) in keys.iter().zip(engine.memory_hashes(&keys)?) {
            ensure!(
                actual == pos::memory_hash(key),
                "OpenCL memhash self-test failed"
            );
        }
        Ok(Self { engine })
    }
    pub fn device(&self) -> &DeviceInfo {
        &self.engine.device
    }

    /// Bounded host batches; the shared engine additionally limits its reusable
    /// device scratch to 4096 keys / 16 MiB, independent of plot ksize.
    pub fn compute_f1(
        &mut self,
        first: u32,
        count: usize,
        id: &Hash,
        ksize: u8,
    ) -> Result<Vec<(u32, [u32; pos::N_META])>> {
        ensure!((8..=32).contains(&ksize), "ksize must be between 8 and 32");
        ensure!(count <= 65536, "OpenCL F1 batch exceeds 65536 entries");
        ensure!(
            first as u64 + count as u64 <= 1u64 << ksize,
            "OpenCL F1 batch exceeds X domain"
        );
        let keys: Vec<_> = (0..count)
            .into_par_iter()
            .with_min_len(256)
            .map(|offset| pos::f1_key(first + offset as u32, id))
            .collect();
        let memory = self
            .engine
            .memory_hashes(&keys)
            .context("OpenCL memhash batch")?;
        keys.par_iter()
            .zip(&memory)
            .with_min_len(256)
            .map(|(key, memory)| Ok(pos::f1_from_memory(key, memory, ksize)?))
            .collect()
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
