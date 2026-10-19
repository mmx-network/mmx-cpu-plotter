# MMX CPU plotter

A parallel Rust CPU plotter for native MMX C0 plots. Supports k29–k32, HDD plots
with metadata, SSD plots without metadata, OG identities and Plot NFT contracts.
The output uses the native VNX `mmx.PlotHeader` and the existing MMX park format.

## Build

Requires Rust 1.89+ and a 64-bit system. Linux supports direct I/O; other systems
use buffered file I/O. The implementation is validated on Linux.

```sh
git submodule update --init --recursive
cargo build --release --locked
```

For a binary optimized for the CPU on the build machine:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release --locked
```

## Plot

Supply the wallet's **33-byte compressed farmer public key**, not a private key.
The output and temporary directories must already exist.

```sh
# Disk mode, HDD plot, direct I/O enabled by default
./target/release/mmx-cpu-plotter \
  --farmer-key YOUR_66_DIGIT_HEX_KEY \
  --ksize 29 --mode disk --threads 16 \
  --temp-dir /fast-temp --output /plots

# Full RAM mode, SSD plot, Plot NFT contract
./target/release/mmx-cpu-plotter \
  --farmer-key YOUR_66_DIGIT_HEX_KEY \
  --contract mmx1YOUR_CONTRACT_ADDRESS \
  --ksize 29 --mode ram --ssd --output /ssd-plots
```

`--mode disk` is the default. It uses external merge sorting with a fixed
64 MiB run buffer and at most 32 merge inputs. Temporary records are streamed
through files and deleted when no longer needed. There is no user RAM-size
setting. The pruning phase also keeps two live-entry bitmaps and a rank index;
their memory scales with ksize, approximately 144 MiB at k29 or 1.1 GiB at k32
when tables have about `2^k` entries. This is additional to I/O buffers, worker
stacks and the OS's own memory use.

`--mode ram` keeps all temporary tables and sorting allocations in RAM and ignores
`--temp-dir`. Sorted tables are read directly from their original allocations.
Matching partitions the exact output into disjoint worker ranges. Disk mode uses
the same parallel matching on bounded input chunks, carrying a complete Y group
across each boundary so matches and parent indices remain identical.
Pruning also streams bounded batches: workers mark shared parent bitmaps with
atomic bit operations and remap live pointers in parallel, preserving table order.
RAM mode requires substantially more RAM; it does not spill to disk. Temporary
metadata records use 80 bytes per entry in RAM, and retained backtrace records
use another 16 bytes per entry per table. Both the current and next tables can
be present during matching. Plan for well over 100 GiB of RAM for k29. On dualv4,
k29 runs with 64 threads peaked at about 137 GiB in RAM mode and 170–178 MiB in
disk mode. HDD plotting took 26m30s in RAM or 97m43s on disk, with direct I/O
enabled. See the [production measurements](docs/dualv4.md) for all four HDD/SSD
results and their limits. Larger plots have not been measured end-to-end, and
peak temporary disk space was not measured. Disk mode needs substantial temporary
space because full metadata records use 76 bytes per entry.

`--direct-io` defaults to enabled. On Linux, temporary file reads/writes and final
plot writes use `O_DIRECT`, 4 KiB aligned buffers and 1 MiB transfers. Final
partial blocks are padded for the transfer and truncated to the exact logical
length. Unsupported filesystems fall back to buffered I/O with a diagnostic.
Use `--direct-io=false` to disable it. RAM mode still uses this flag for the final
plot. Device and filesystem caches beyond the OS page cache remain independent.

OpenCL can offload **only table-1 memhash** to a GPU. All SHA-512 hashing,
matching, sorting and pruning use the CPU. Enable it explicitly:

```sh
./target/release/mmx-cpu-plotter --list-opencl-devices
./target/release/mmx-cpu-plotter --farmer-key YOUR_66_DIGIT_HEX_KEY \
  --ksize 29 --mode ram --threads 16 --opencl --opencl-device 0 --output /plots
```

The shared `mmx-opencl` implementation comes from the pinned
`vendor/mmx-rust-node` submodule. CPU operation needs no OpenCL installation;
GPU operation requires an OpenCL 1.2 GPU driver and runtime loader. Device listing
and selection include only devices reporting `CL_DEVICE_TYPE_GPU`; CPU OpenCL
implementations are excluded. An explicitly requested GPU must initialize and pass a CPU comparison before plotting starts;
errors abort without publishing a plot. Omitting `--opencl` selects the CPU path.
The backend pipelines three streams through separate OpenCL queues
with reusable device buffers and pinned host storage. CPU SHA-512 work,
transfers, GPU kernels and consumption by the sorter overlap across batches;
completion events guard buffer reuse. Supported NVIDIA GPUs use warp shuffles
and 65,536-entry streams, following the CUDA plotter's memhash layout. These use
768 MiB of device scratch in total, plus 36 MiB each of device I/O buffers and
pinned host storage. Smaller NVIDIA GPUs use 16,384 entries per stream; other
GPUs use the portable kernel with 4,096 entries per stream.
GPU resources are released after T1, before subsequent table matching.
Driver/context memory is additional.
See [OpenCL measurements and validation](docs/opencl.md).

Other options:

- `--count N`: create N plots, each with a fresh OS-random 32-byte seed.
- `--seed HEX`: reproducible seed, permitted only with `--count 1`.
- `--contract`: accepts an MMX Bech32m address or 32 native address bytes in hex.
- `--ssd`: omit metadata; farming these plots requires SSD random I/O performance.
- `--allow-small-k`: permit k8–k28 for development. These plots cannot farm mainnet.

Completed plots are synced and published without overwriting an existing file.
Errors and cooperative Ctrl-C cancellation remove this run's temporary data and
incomplete output. A hard kill or machine failure can leave temporary files;
`.mmx-plot-*` directories belong to individual disk-mode runs. There is no resume
support. Very small development plots can have too few matches, excessive live
pointer counts, or deltas outside the native 0–47 range; those fail explicitly
without publishing an invalid plot. Choose another seed in that case.

## Verify and measure

```sh
cargo test --all-targets --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --all --check

# Development benchmark: both modes, direct I/O on/off, elapsed time and peak RSS.
# Checks that every resulting plot has the same SHA-256; cleans up its plots.
python3 tools/bench_plot.py --ksize 20 --threads 4 --rounds 2

# Isolate CPU leaf hashing from sorting and disk I/O (count, threads).
cargo run --release --example bench_f1 -- 200000 64

# Compare full F1 work with CPU-only versus GPU memhash (count, threads, GPU).
cargo run --release --locked --example bench_memhash -- 2000000 64 0

# Require real GPU tests; compares CPU/GPU output, including complete plots.
MMX_OPENCL_TESTS=1 cargo test --all-targets --locked

# Optional independent native reader/qualities/full CPU proof check.
# Requires an existing mmx-node CMake build, with test_pos_compute built.
python3 tools/check_native.py --source ../mmx-node /plots/example.plot --samples 32

# Explicit production test matrix. Retains four plots, logs, timing/RSS data,
# checksums and native verification results in a new output directory.
python3 tools/test_matrix.py --ksize 29 --threads 32 \
  --output /plots/test-matrix --native-source ../mmx-node
# Add --opencl --opencl-device 0 to test GPU memhash in all four cases.
```

An earlier local release-build run at k20 with four worker threads produced
identical 52,179,044-byte plots in every combination:

| Storage | Direct I/O | Seconds | Peak RSS |
| --- | --- | ---: | ---: |
| RAM | off | 4.59 | 271 MiB |
| RAM | on | 4.77 | 271 MiB |
| Disk | off | 5.94 | 71 MiB |
| Disk | on | 7.83 | 71 MiB |

The direct-I/O disk path improved from 11.19 to 7.83 seconds after increasing
transfer sizes and extracting backtraces during matching. These are small-plot
smoke measurements on one machine, with one run per configuration. Direct I/O's
purpose is to avoid displacing useful OS cache contents; it can be slower than
buffered I/O, particularly when the working set fits in the page cache.

Tests compare native C++ leaf/memory-hash vectors at k8/k16/k29/k32, exact native
header identities, complete small-plot Y/metadata tables, reconstructed 256-X
proofs, RAM/disk determinism, multi-pass merge sorting, direct-I/O alignment and
tail handling, overwrite protection and cancellation. Production k29 HDD and SSD
plots in both storage modes passed the native reader, quality lookup and full
CPU verifier on 64 sampled proofs per plot. SHA-256 digests match between RAM
and disk modes for each format. See the [dualv4 report](docs/dualv4.md) for details.
k30–k32 have not been plotted end-to-end.

The hashing/proof reference is `mmx-rust-node` and `mmx-node` revision
`ca6438b37c08007106d58efe4a990f6ecb276649`. `src/pos.rs` and
`schema/PlotHeader.vni` come from the Rust node; the three external header types
are declared in `build.rs`. The VNX Rust dependency is pinned as a submodule.
Licensed under Apache-2.0; see [LICENSE](LICENSE).
