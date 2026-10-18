`leaves.jsonl` contains the 32 native C++ F1/memory-hash vectors from
`mmx-rust-node/crates/mmx-consensus/tests/fixtures/work.jsonl`. They cover k8,
k16, k29, and k32. The oracle uses mmx-node revision
`ca6438b37c08007106d58efe4a990f6ecb276649`; the original generator is
`mmx-rust-node/tools/generate_work_fixtures.py`.

These compare the 4 KiB memory hash and leaf metadata without requiring a C++
checkout. `tools/check_native.py` additionally runs the C++ reader and verifier
against completed plots when a built mmx-node checkout is available.
