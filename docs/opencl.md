# OpenCL memhash

Only the 4 KiB memory hash in table 1 runs on the GPU. Both SHA-512 steps around
it remain CPU operations. Tables 2–9, sorting and pruning use the CPU, and the
plot format is unchanged.

```sh
./target/release/mmx-cpu-plotter --list-opencl-devices
./target/release/mmx-cpu-plotter --farmer-key YOUR_66_DIGIT_HEX_KEY \
  --mode ram --ksize 29 --threads 16 --opencl --opencl-device 0 --output /plots
```

`--opencl` is opt-in; its default device is 0. The shared `mmx-opencl` backend is
pinned through `vendor/mmx-rust-node`. CPU operation needs no OpenCL SDK or runtime;
GPU operation requires an OpenCL 1.2 GPU driver/runtime. Listing and selection
filter for `CL_DEVICE_TYPE_GPU` and verify each returned device's type. CPU OpenCL
implementations are excluded, with no fallback to a CPU OpenCL device. Device
indices refer to the GPU-only list. The requested GPU must
initialize and match CPU test keys before temporary plotting data is created.
Failures abort the run and clean up temporary data and incomplete output.

## Pipeline

Three independent in-order OpenCL command queues operate on three reusable
buffer sets. Every stream queues nonblocking upload → memory generation →
memory mixing → nonblocking download, then flushes immediately. Queue ordering
supplies the dependencies within each stream. Each completion event guards
retirement and reuse of that stream's buffers.

CPU workers write SHA-512 keys directly into pinned staging storage while older
batches run on the GPU. After retiring the oldest batch, workers finish its
SHA-512 metadata and pass ordered results to the sorter while later batches
remain in flight. The pipeline continues across the entire T1 domain; it does
not drain at the former 65,536-entry host batch boundaries. Ordinary steady-state
work uses no blocking transfers or `clFinish` calls. Cancellation and callback/I/O
errors drain all pending queues before returning; destruction also drains before
unmapping or releasing buffers.

The NVIDIA fast path follows the execution layout in
[`Node_phase1.cu`](https://github.com/madMAx43v3r/mmx-cuda-plotter/blob/b03a655c6b0e985fb51fceb202d4b8973439eccd/src/Node_phase1.cu#L83).
Each 128-thread workgroup processes four independent hashes, one per 32-lane
warp, with 16 KiB of shared scratch. Inline PTX shuffles replace the portable
kernel's shared-memory reduction, and warp barriers replace its workgroup
barriers. Each iteration needs two warp barriers instead of eight workgroup
barriers. Each lane writes a distinct scratch word, so warp ordering also permits
plain XOR writes instead of atomics. Memory generation uses 256-thread groups
and 16-byte vector stores. This remains OpenCL; a CUDA runtime is not required.

Selection requires NVIDIA's vendor ID, a reported 32-lane warp and compute
capability 7 or newer. Other GPUs keep the portable OpenCL kernel. Both paths
must pass the CPU self-test. No standard OpenCL subgroup extension is required.
The CUDA plotter also runs both SHA-512 stages on its GPU; this implementation
keeps them on the CPU as requested.

Like the CUDA plotter, NVIDIA GPUs with at least 4 GiB of global memory use
65,536 entries per stream. Smaller NVIDIA GPUs use 16,384; the portable path
retains 4,096. Resource totals across all three streams are:

| Entries per stream | Device scratch | Device I/O buffers | Pinned host storage |
| ---: | ---: | ---: | ---: |
| 65,536 | 768 MiB | 36 MiB | 36 MiB |
| 16,384 | 192 MiB | 9 MiB | 9 MiB |
| 4,096 | 48 MiB | 2.25 MiB | 2.25 MiB |

CPU finalization reuses one batch buffer. Memory remains bounded regardless of
ksize; driver/context allocations are additional. GPU resources are released
after T1 generation. The same path supports RAM/disk modes, HDD/SSD formats and
direct I/O.

## Measurements

On 2026-10-19, `bench_memhash` compared every complete F1 result with the CPU
reference. Times include CPU SHA-512, transfers and result collection. Its
blocking baseline uses the portable kernel, vector stores and the previous GPU
execution order. Three timed rounds followed warmup:

| Machine / GPU | CPU threads | F1 values | CPU X/sec | Portable blocking X/sec | NVIDIA pipeline X/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core Ultra 7 265KF / RTX 4070 SUPER | 4 | 1,000,000 | 0.408M–0.428M | 1.96M–2.92M | 6.13M–6.61M |
| dualv4 / RTX 4060 Ti | 64 | 2,000,000 | 1.55M–1.58M | 1.96M–1.98M | 6.35M–6.49M |

Relative to CPU-only F1 computation, the pipeline was 15.05–15.49× faster locally
and 4.10–4.13× faster on dualv4. Before the warp kernel and larger tiles, the
4060 Ti's [portable pipeline](opencl-pipeline.json) reached 2.22M–2.36M X/sec;
the new path is about 2.8× faster. The local binary uses the generic release CPU
target; dualv4 uses `RUSTFLAGS="-C target-cpu=native"`. Both use k29 F1 with an
all-zero plot ID. These shared machines were not isolated, and local blocking
times varied considerably. All rounds are retained. These are F1 measurements,
not whole-plot production speedups or comparisons with a running CUDA plotter.

An additional profiled run verified actual cross-queue overlap from device event
timestamps. For 16 batches, the sum of pairwise kernel overlaps was 1.166 ms on
the 4070 SUPER and 38.973 ms on the 4060 Ti; copy/compute overlaps were 0.495 ms
and 30.813 ms respectively. Pairwise sums can count three-way overlaps more than
once and are not utilization percentages. Profiling is disabled during normal
plotting and recorded outside timed benchmark rounds. The benchmark also prints
mean upload, generation, mixing and download times; concurrent queues can affect
these stage durations.

Both GPUs passed complete k20 RAM/disk × HDD/SSD matrices with direct I/O enabled.
Every file matched the earlier CPU plot's SHA-256 digest for its format. The
native C++ reader, quality lookup and full CPU verifier passed 16 sampled 256-X
proofs per file. Checksums:

- HDD: `7542d96f65625a229d4663e8d930418decc431c381dcf18b260d66b61706c59c`
- SSD: `ba59e344a428495781c4d3f2ed02731c6e3a91f2de75bf4949438a290de1a4c1`

Real-device tests cover all three streams, partial batches, multiple ring wraps,
X-domain limits through k32, ordered output across the old host boundary,
cancellation/drain/reuse and destruction with outstanding DMA. Both fast and
portable kernels are compared with CPU results, including batches with a
partially occupied final workgroup. Shared-backend tests additionally retire
streams out of order and reject premature reuse.
Full k29 GPU plotting has not been tested; the [k29 production matrix](dualv4.md)
used CPU hashing.

Current measurements are in [opencl-warp.json](opencl-warp.json); the earlier
[portable pipeline](opencl-pipeline.json) and
[initial blocking-backend results](opencl-memhash.json) are retained separately.
The dualv4 artifacts are in
`/home/mad/mmx-cpu-plotter-tests/opencl-warp-k20`.

## Reproduce

```sh
cargo build --release --locked
MMX_OPENCL_TESTS=1 cargo test --all-targets --locked
MMX_OPENCL_TESTS=1 cargo test --manifest-path vendor/mmx-rust-node/Cargo.toml \
  -p mmx-opencl --locked
cargo run --release --locked --example bench_memhash -- 2000000 64 0
python3 tools/test_matrix.py --ksize 20 --threads 64 --opencl --opencl-device 0 \
  --output /plots/gpu-pipeline-test --native-source ../mmx-node --samples 16
```
