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
This mode requires substantially more RAM; it does not spill to disk. Temporary
metadata records use 80 bytes per entry in RAM, and retained backtrace records
use another 16 bytes per entry per table. Both the current and next tables can
be present during matching. Plan for well over 100 GiB of RAM for k29. Production
peak RAM, disk space and plotting times have not been benchmarked; small-plot
measurements are not production sizing guarantees. Disk mode likewise needs
substantial temporary space because full metadata records use 76 bytes per entry.

`--direct-io` defaults to enabled. On Linux, temporary file reads/writes and final
plot writes use `O_DIRECT`, 4 KiB aligned buffers and 1 MiB transfers. Final
partial blocks are padded for the transfer and truncated to the exact logical
length. Unsupported filesystems fall back to buffered I/O with a diagnostic.
Use `--direct-io=false` to disable it. RAM mode still uses this flag for the final
plot. Device and filesystem caches beyond the OS page cache remain independent.

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

# Optional independent native reader/qualities/full CPU proof check.
# Requires an existing mmx-node CMake build, with test_pos_compute built.
python3 tools/check_native.py --source ../mmx-node /plots/example.plot --samples 32

# Explicit production test matrix. Retains four plots, logs, timing/RSS data,
# checksums and native verification results in a new output directory.
python3 tools/test_matrix.py --ksize 29 --threads 32 \
  --output /plots/test-matrix --native-source ../mmx-node
```

One local release-build run at k20 with four worker threads produced identical
52,179,044-byte plots in every combination:

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
tail handling, overwrite protection and cancellation. End-to-end native testing
uses development plots; a production-sized k29–k32 plot has not been generated
as part of this implementation's validation.

The hashing/proof reference is `mmx-rust-node` and `mmx-node` revision
`ca6438b37c08007106d58efe4a990f6ecb276649`. `src/pos.rs` and
`schema/PlotHeader.vni` come from the Rust node; the three external header types
are declared in `build.rs`. The VNX Rust dependency is pinned as a submodule.
Licensed under Apache-2.0; see [LICENSE](LICENSE).
