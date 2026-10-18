#!/usr/bin/env python3
"""Measure both storage modes with direct I/O on/off, and compare output hashes.

Uses small development plots by default; never runs a production k29+ plot.
Requires GNU time. Temporary plots are removed after each measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--binary", type=Path, default=Path("target/release/mmx-cpu-plotter"))
p.add_argument("--ksize", type=int, default=20, choices=range(12, 25))
p.add_argument("--threads", type=int, default=4)
p.add_argument("--rounds", type=int, default=1)
p.add_argument("--temp-dir", type=Path)
a = p.parse_args()
binary = a.binary.resolve()
expected = None
for mode in ("ram", "disk"):
    for direct in (False, True):
        for iteration in range(a.rounds):
            with tempfile.TemporaryDirectory(prefix="mmx-plot-bench-", dir=a.temp_dir) as tmp:
                directory = Path(tmp)
                stats = directory / "time.json"
                command = [str(binary), "--farmer-key", "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798", "--ksize", str(a.ksize), "--allow-small-k", "--seed", "00" * 32, "--mode", mode, f"--direct-io={str(direct).lower()}", "--threads", str(a.threads), "--output", tmp]
                result = subprocess.run(["/usr/bin/time", "-f", '{"seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M}', "-o", str(stats), *command], capture_output=True, text=True)
                if result.returncode:
                    raise SystemExit(result.stderr)
                plot = next(directory.glob("*.plot"))
                with plot.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if expected is not None and digest != expected:
                    raise SystemExit("output changed between storage/I/O modes")
                expected = digest
                print(json.dumps({"mode": mode, "direct_io": direct, "round": iteration + 1, "ksize": a.ksize, "threads": a.threads, **json.loads(stats.read_text()), "plot_bytes": plot.stat().st_size, "sha256": digest}), flush=True)
