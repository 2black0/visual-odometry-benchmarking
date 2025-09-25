#!/usr/bin/env python3
import sys, re, numpy as np
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python to_kitti_kf.py <in_xyz.txt> <out_kitti.txt>")
    sys.exit(1)

inp, outp = Path(sys.argv[1]), Path(sys.argv[2])
out_lines = []
with inp.open("r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # ambil 3 angka pertama sebagai x y z (toleran koma/tab)
        toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(toks) < 3: 
            continue
        x, y, z = map(float, toks[:3])
        M = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z]], float).reshape(-1)
        out_lines.append(" ".join(f"{v:.9e}" for v in M))
outp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
print(f"OK: wrote {len(out_lines)} poses to {outp}")
