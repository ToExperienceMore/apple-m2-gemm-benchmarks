"""Summarize a completed reproduction without changing published results."""

import argparse, csv, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
a = p.parse_args()
d = a.run.resolve()
if not (d / "VALID.json").exists():
    p.error("Run is incomplete: no VALID.json")
rows = []
if (d / "all.json").exists():
    for x in json.loads((d / "all.json").read_text()):
        rows.append(
            dict(
                n=x["n"],
                backend=x["backend"],
                threads=x.get("threads", ""),
                mean_ms=x["mean_ms"],
                tflops=x["tflops"],
                relative_l2=x.get("relative_l2", x.get("relative_l2_full", "")),
            )
        )
        if "prepacked_ms" in x:
            rows.append(
                dict(
                    n=x["n"],
                    backend="amx-fp16-prepacked",
                    threads=x["threads"],
                    mean_ms=x["prepacked_ms"],
                    tflops=x["prepacked_tflops"],
                    relative_l2=x["relative_l2"],
                )
            )
lines = [
    "# Reproduction results",
    "",
    "All times are mean milliseconds per complete call. AMX prepacked excludes A/B packing.",
    "",
    "| N | Backend | Threads | Mean ms | TFLOPS | Relative L2 |",
    "|---:|---|---:|---:|---:|---:|",
]
for x in rows:
    lines.append(
        f"| {x['n']} | {x['backend']} | {x['threads']} | {x['mean_ms']:.6f} | {x['tflops']:.6f} | {x['relative_l2']} |"
    )
if rows:
    with (d / "summary.csv").open("w") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
for path in sorted(d.glob("amx-instructions-*.csv")):
    lines += [
        "",
        f"## {path.name}",
        "",
        "| Mode | Z accumulators | TFLOPS |",
        "|---|---:|---:|",
    ]
    for x in csv.DictReader(path.open()):
        lines.append(
            f"| {x['test_name']} | {x['num_z']} | {int(x['ops_total'])/int(x['ns_elapsed'])/1000:.6f} |"
        )
(d / "summary.md").write_text("\n".join(lines) + "\n")
print(d / "summary.md")
