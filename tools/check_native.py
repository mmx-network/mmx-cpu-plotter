#!/usr/bin/env python3
"""Compile a small oracle using an existing mmx-node build, then check plots."""
import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("plots", nargs="+", type=Path)
p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "mmx-node")
p.add_argument("--build", type=Path)
p.add_argument("--samples", type=int, default=16)
a = p.parse_args()
build = (a.build or a.source / "build").resolve()
target = build / "test/CMakeFiles/test_pos_compute.dir"
flags = {}
for line in (target / "flags.make").read_text().splitlines():
    if " = " in line:
        key, value = line.split(" = ", 1)
        flags[key] = shlex.split(value)
link = shlex.split((target / "link.txt").read_text())
with tempfile.TemporaryDirectory(prefix="mmx-plot-oracle-") as tmp:
    binary = Path(tmp) / "check_native"
    command = [link[0], "-std=c++17", "-O2", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"], str(Path(__file__).with_suffix(".cpp").resolve())]
    i = 1
    while i < len(link):
        arg = link[i]
        if arg == "-o":
            i += 2
            continue
        if not arg.endswith("test_pos_compute.cpp.o"):
            command.append(arg)
        i += 1
    subprocess.run([*command, "-o", str(binary)], cwd=build / "test", check=True)
    for plot in a.plots:
        subprocess.run([str(binary), str(plot.resolve()), str(a.samples)], check=True)
