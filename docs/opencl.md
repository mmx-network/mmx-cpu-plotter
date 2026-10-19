# OpenCL memhash

Only the 4 KiB memory hash in table 1 runs on the GPU. The SHA-512 that derives
its 64-byte key and the SHA-512 that turns `key || memory_hash` into F1 metadata
remain parallel CPU operations. Tables 2–9, sorting and pruning use the CPU.
The plot format is unchanged.

The plotter uses `mmx-opencl` from the `mmx-rust-node` submodule, pinned at
`26d0122ee015ac32651676b3ed9cf20934e4aef4`. It uses a runtime-loaded OpenCL 1.2
interface, so the CPU path needs no OpenCL SDK or runtime. GPU operation requires
a working GPU ICD/driver. Devices are listed in GPU discovery order:

```sh
./target/release/mmx-cpu-plotter --list-opencl-devices
./target/release/mmx-cpu-plotter --farmer-key YOUR_66_DIGIT_HEX_KEY \
  --mode ram --ksize 29 --threads 16 --opencl --opencl-device 0 --output /plots
```

`--opencl` is opt-in; its default device is 0. An explicitly requested GPU must
initialize and match CPU results for three test keys before temporary plotting
data is created. Initialization, compilation or execution failures stop the run;
normal error cleanup removes temporary data and incomplete output. Omit the flag
to plot entirely on the CPU.

Host work is bounded to 65,536 F1 entries per batch. The shared engine processes
4,096 keys at a time, reusing 16 MiB of device scratch, a 256 KiB input buffer and
a 512 KiB output buffer. Driver/context memory is additional. GPU resources are
released after T1 generation, before sorting its result and matching later tables.
The same path supports RAM/disk modes, HDD/SSD formats and direct I/O.

## Measurements

On 2026-10-19, `bench_memhash` compared every complete F1 result with the CPU
implementation, including all CPU SHA-512 work and GPU transfers in its timing.
Each row covers three timed rounds after warmup:

| Machine / GPU | CPU threads | F1 values | CPU X/sec | GPU memhash X/sec | F1 speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core Ultra 7 265KF / RTX 4070 SUPER | 4 | 200,000 | 421k–429k | 2.16M–2.77M | 5.02–6.57× |
| dualv4 / RTX 4060 Ti | 64 | 2,000,000 | 1.54M–1.59M | 1.59M–1.71M | 1.03–1.11× |

Both use k29 F1 with an all-zero plot ID, independent of plot identity generation.
The local binary uses the generic release CPU target; dualv4 uses
`RUSTFLAGS="-C target-cpu=native"`. These shared machines were not isolated.
These numbers measure F1 throughput, not overall plotting speed. On a large CPU,
GPU offload can offer a smaller throughput benefit while freeing CPU time.

A local k20 smoke matrix used four threads and direct I/O. Each cell is one run:

| Storage | Format | CPU only | GPU memhash |
| --- | --- | ---: | ---: |
| RAM | HDD | 4.00 s | 2.21 s |
| RAM | SSD | 3.93 s | 2.05 s |
| Disk | HDD | 6.58 s | 4.89 s |
| Disk | SSD | 6.40 s | 5.45 s |

GPU runs peaked at about 308 MiB RSS in RAM mode and 192 MiB in disk mode on the
local machine, including driver allocations. The 4060 Ti also passed the full
k20 matrix on dualv4 with 64 CPU threads. All GPU plots have exactly the same
SHA-256 digests as CPU plots in their respective formats:

- HDD: `7542d96f65625a229d4663e8d930418decc431c381dcf18b260d66b61706c59c`
- SSD: `ba59e344a428495781c4d3f2ed02731c6e3a91f2de75bf4949438a290de1a4c1`

The native C++ reader, quality lookup and full CPU verifier passed 16 sampled
256-X proofs per file for the CPU/local GPU/dualv4 GPU matrices. Real-device
unit tests additionally cover scratch-batch boundaries, buffer reuse, k8/k16/k29/
k32 and the final X values at each domain boundary. CPU-only plotting was also
checked with no installed ICDs visible. Full k29 GPU plotting has not been tested;
the [earlier k29 production matrix](dualv4.md) used CPU hashing.

Raw timings, checksums, binary digests and native source revisions are recorded
in [opencl-memhash.json](opencl-memhash.json). The dualv4 artifacts are retained in
`/home/mad/mmx-cpu-plotter-tests/opencl-memhash-k20`.

## Reproduce

```sh
cargo build --release --locked
MMX_OPENCL_TESTS=1 cargo test --all-targets --locked
cargo run --release --locked --example bench_memhash -- 2000000 64 0
python3 tools/test_matrix.py --ksize 20 --threads 64 --opencl --opencl-device 0 \
  --output /plots/gpu-memhash-test --native-source ../mmx-node --samples 16
```
