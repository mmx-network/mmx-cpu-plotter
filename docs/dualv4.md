# dualv4 production validation

Measurements from 2026-10-18 on dualv4: two Xeon E5-2683 v4 CPUs,
32 physical cores / 64 logical CPUs, 256 GiB installed RAM (251.6 GiB usable),
and an ext4 filesystem on NVMe. Every plot uses k29, 64 worker threads and
direct I/O enabled. HDD and SSD refer to the **plot formats**; both were
generated on NVMe, not separate HDD and SSD devices.

The release binary was built with Rust 1.97.0 and
`RUSTFLAGS="-C target-cpu=native" cargo build --release --locked` on
Linux 6.8.0-138-generic. The tested implementation is
`f5aa994b0051b4b17ed0a6a87f813f065b3ef3fc`; its binary SHA-256 is
`35f1cfece6e0ea3ea3e4a4ef848a1852c41c4374e4c8098eddc923cd5adacb9f`.

## Reproduce

```sh
python3 tools/test_matrix.py --ksize 29 --threads 64 \
  --output /home/mad/mmx-cpu-plotter-tests/k29-f5aa994 \
  --native-source /home/mad/mmx-node --samples 64
```

The runner uses the public test farmer key
`0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798`
and an all-zero seed. These plots are compatibility artifacts, not plots for a
user's wallet. Their plot ID is
`682a7fb5e4e712d6b4263052d99529ccd3d87a5d602b1535539c00fd0dcd8a3a`.
Logs, plots and machine-readable results are retained in the output directory.
Wall time and peak process RSS come from GNU time; checksum and subsequent
native verification time are excluded from plotting time.

## Results

All four cases passed. Each configuration was run once, in the order below.

| Storage mode | Plot format | Plotting time | Peak RSS | Plot size |
| --- | --- | ---: | ---: | ---: |
| RAM | HDD | 26m30s | 136.55 GiB | 36.60 GiB |
| RAM | SSD | 24m45s | 136.54 GiB | 14.67 GiB |
| Disk | HDD | 97m43s | 169.8 MiB | 36.60 GiB |
| Disk | SSD | 95m14s | 177.2 MiB | 14.67 GiB |

The HDD files are byte-for-byte identical between storage modes, as are the SSD
files. HDD plots contain 39,295,838,764 bytes; SSD plots contain 15,756,205,612
bytes. Each contains 541,139,108 final entries. The independent native C++ reader,
quality lookup and full CPU verifier passed 64 sampled proofs per plot (256
samples total), using the existing `mmx-node` build at revision
`41f2e4ed205dfad9e64081e3501547d6db54cb52`.

There were no direct-I/O fallback messages, and all plotting temporary files
were removed after completion. Peak temporary disk usage was not measured.
These k29 measurements do not establish sizing or compatibility for k30–k32;
those sizes have not been plotted end-to-end. Exact timings, CPU time, RSS,
SHA-256 digests, build details and native output are in the
[machine-readable results](dualv4-k29.json).

## Pruning improvement

Pruning walks backwards through the tables, keeping entries referenced by the
next table. Each table requires a parent-marking pass, a bitmap rank index, and
a second pass to replace old parent positions with compact positions. The
original implementation performed parent marking and remapping serially.
At k29, each pass visits hundreds of millions of pairs.

The optimized version reads batches of 65,536 pairs (1 MiB). Workers mark a
shared bitmap using atomic OR, combining both parent bits into one operation
when they share a word. After marking completes, the bitmap becomes an ordinary
read-only rank index. Workers then filter live entries and remap their parent
pointers in parallel. Collection preserves input order, and a buffered writer
emits the resulting pairs in that order. Batches keep working memory bounded
in both storage modes; the optimization does not add full-table worker copies.

| RAM/HDD implementation | Plotting time | Peak RSS |
| --- | ---: | ---: |
| Before parallel pruning (`ff8208a`) | 2,392.90 s (39m53s) | 136.55 GiB |
| Parallel pruning (`f5aa994`) | 1,589.57 s (26m30s) | 136.55 GiB |

This single-run comparison reduced total plotting time by 33.6% (1.51x
throughput). Both files contain exactly 39,295,838,764 bytes and have SHA-256
`e76390d59a58bd6b92f252b71e59b876f642fd79a24814c4860d179cbc06cdfc`.
All live-entry counts also match. The baseline plot passed 256 sampled native
checks. Its final pruning/writing overlapped the optimized run's initial leaf
hashing, so these measurements are not fully isolated repetitions.

The shared rank structure and output stream still impose serial work. The
reported reduction is measured end-to-end, not a claim that every pruning
operation scales linearly with thread count.
