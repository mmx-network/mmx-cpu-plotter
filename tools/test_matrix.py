#!/usr/bin/env python3
"""Run HDD/SSD plots in both storage modes and retain timings and verification.

Production sizes are intentionally explicit. All runs use a public test identity,
so the generated plots are compatibility artifacts, not farmable wallet plots.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

KEY = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"


def now():
    return datetime.now(timezone.utc).isoformat()


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            # Verification should not leave whole production plots in page cache.
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(stream.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/mmx-cpu-plotter"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ksize", type=int, choices=range(8, 33), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--native-source", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    if args.threads < 1 or args.samples < 1:
        parser.error("threads and samples must be positive")
    binary = args.binary.resolve()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "results.json"
    if report_path.exists():
        parser.error("output already contains a report; choose a fresh directory")
    repository = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    report = {"started": now(), "revision": revision, "binary_sha256": checksum(binary), "ksize": args.ksize, "threads": args.threads, "direct_io": True, "test_farmer_key": KEY, "seed": "00" * 32, "cases": [], "status": "running"}
    report["hardware"] = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))

    def save():
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(report_path)

    save()
    try:
        digests = {}
        plots = []
        for mode in ("ram", "disk"):
            for kind in ("hdd", "ssd"):
                directory = root / (mode + "-" + kind)
                directory.mkdir()
                case = {"mode": mode, "type": kind, "started": now(), "status": "plotting", "log": str(directory / "plot.log")}
                report["cases"].append(case)
                save()
                command = [str(binary), "--farmer-key", KEY, "--ksize", str(args.ksize), "--seed", "00" * 32, "--mode", mode, "--threads", str(args.threads), "--output", str(directory)]
                if args.ksize < 29:
                    command.append("--allow-small-k")
                if kind == "ssd":
                    command.append("--ssd")
                case["command"] = command
                print(now(), mode, kind, "starting", flush=True)
                with (directory / "plot.log").open("w") as log:
                    result = subprocess.run(["/usr/bin/time", "-f", '{"seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M}', "-o", str(directory / "time.json"), *command], stdout=log, stderr=subprocess.STDOUT)
                case["returncode"] = result.returncode
                case["finished_plotting"] = now()
                if result.returncode:
                    raise RuntimeError("plot failed: " + str(directory / "plot.log"))
                case.update(json.loads((directory / "time.json").read_text()))
                plot = next(directory.glob("*.plot"))
                case.update({"path": str(plot), "plot_bytes": plot.stat().st_size, "status": "checksumming"})
                save()
                began = time.monotonic()
                digest = checksum(plot)
                case.update({"sha256": digest, "checksum_seconds": time.monotonic() - began})
                if kind in digests and digests[kind] != digest:
                    raise RuntimeError(kind + " plot differs between RAM/disk modes")
                digests[kind] = digest
                case["status"] = "passed"
                plots.append(plot)
                save()
                print(now(), mode, kind, "passed", case["seconds"], "seconds", case["max_rss_kib"], "KiB peak RSS", flush=True)
        if args.native_source:
            report["status"] = "native_verification"
            save()
            command = [sys.executable, str(repository / "tools/check_native.py"), "--source", str(args.native_source.resolve()), "--samples", str(args.samples), *map(str, plots)]
            with (root / "native.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            report["native_samples_per_plot"] = args.samples
            report["native_source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.native_source, text=True).strip()
        report.update({"status": "passed", "finished": now()})
        save()
        print(now(), "all cases passed", flush=True)
    except BaseException as error:
        report.update({"status": "failed", "error": str(error), "finished": now()})
        save()
        raise


if __name__ == "__main__":
    main()
