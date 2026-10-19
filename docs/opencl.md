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
GPU operation requires an OpenCL 1.2 GPU driver/runtime. The requested GPU must
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

Each stream holds 4,096 entries, with 16 MiB of device scratch, 256 KiB of input
and 512 KiB of output. Across three streams this totals 50.25 MiB of device
buffers plus 2.25 MiB of pinned host storage. CPU finalization reuses a 4,096-entry
buffer. Memory remains bounded regardless of ksize; driver/context allocations
are additional. GPU resources are released after T1 generation. The same path
supports RAM/disk modes, HDD/SSD formats and direct I/O.

## Measurements

On 2026-10-19, `bench_memhash` compared every complete F1 result with the CPU
reference. Times include CPU SHA-512, transfers and result collection. Its
blocking baseline preserves the previous GPU execution order. Three timed
rounds followed warmup:

| Machine / GPU | CPU threads | F1 values | Blocking GPU X/sec | Pipelined X/sec | Pipeline speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core Ultra 7 265KF / RTX 4070 SUPER | 4 | 1,000,000 | 2.68M–2.79M | 4.62M–4.73M | 1.69–1.77× |
| dualv4 / RTX 4060 Ti | 64 | 2,000,000 | 1.38M–1.73M | 2.22M–2.36M | 1.28–1.70× |

Relative to CPU-only F1 computation, the pipeline was 10.86–11.09× faster locally
and 1.40–1.51× faster on dualv4. The local binary uses the generic release CPU
target; dualv4 uses `RUSTFLAGS="-C target-cpu=native"`. Both use k29 F1 with an
all-zero plot ID. These shared machines were not isolated. The third dualv4 blocking round took
1.4440 s versus about 1.154 s in the first two; all rounds are retained. These are F1
measurements, not whole-plot production speedups.

An additional profiled run verified actual cross-queue overlap from device event
timestamps. For 16 batches, the sum of pairwise kernel overlaps was 13.355 ms on
the 4070 SUPER and 4.346 ms on the 4060 Ti; copy/compute overlaps were 0.967 ms
and 0.634 ms respectively. Pairwise sums can count three-way overlaps more than
once and are not utilization percentages. Profiling is disabled during normal
plotting and recorded outside timed benchmark rounds.

Both GPUs passed complete k20 RAM/disk × HDD/SSD matrices with direct I/O enabled.
Every file matched the earlier CPU plot's SHA-256 digest for its format. The
native C++ reader, quality lookup and full CPU verifier passed 16 sampled 256-X
proofs per file. Checksums:

- HDD: `7542d96f65625a229d4663e8d930418decc431c381dcf18b260d66b61706c59c`
- SSD: `ba59e344a428495781c4d3f2ed02731c6e3a91f2de75bf4949438a290de1a4c1`

Real-device tests cover all three streams, partial batches, multiple ring wraps,
X-domain limits through k32, ordered output across the old host boundary,
cancellation/drain/reuse and destruction with outstanding DMA. Shared-backend
tests additionally retire streams out of order and reject premature reuse.
Full k29 GPU plotting has not been tested; the [k29 production matrix](dualv4.md)
used CPU hashing.

Current measurements are in [opencl-pipeline.json](opencl-pipeline.json); the
[initial blocking-backend results](opencl-memhash.json) are retained separately.
The dualv4 artifacts are in
`/home/mad/mmx-cpu-plotter-tests/opencl-pipeline-final-k20`.

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
