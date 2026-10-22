"""Plot archived LiDAR diagnostics in English without running the model."""
import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, required=True, help="ZIP archive of the historical T4 ablation diagnostics.")
parser.add_argument("--output", type=Path, required=True, help="Output PNG file.")
args = parser.parse_args()
ARCHIVE = args.input
DESTINATION = args.output
paths = ["reference_8x5/LiDAR_1_steps.json", "bank16_steps5/steps.json", "bank8_steps20/steps.json"]
labels = ["8 candidates × 5 steps (reference)", "16 candidates × 5 steps", "8 candidates × 20 steps"]
colors = ["#2269AC", "#21845B", "#C4473A"]
if ARCHIVE.is_dir():
    series = [json.loads((ARCHIVE / path).read_text(encoding='utf-8')) for path in paths]
else:
    with zipfile.ZipFile(ARCHIVE) as archive:
        series = [json.loads(archive.read(next(name for name in archive.namelist()
                                             if name.endswith("/" + path)))) for path in paths]
assert all(len(rows) == 50 for rows in series)
assert all(row["beta"] == series[0][0]["beta"] for rows in series for row in rows)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.subplots_adjust(left=.095, right=.965, bottom=.18, top=.785, hspace=.30)
fig.suptitle("LiDAR: two bank ablations at fixed β", x=.095, y=.972,
             ha="left", fontsize=16, fontweight="bold")
fig.text(.095, .928, "T4 pilot · 1 prompt, 1 seed · β = 66.3703 · 50-step final DDIM generation",
         ha="left", fontsize=10, color="#454D56")
handles = [Line2D([0], [0], color=color, lw=2.3, label=label)
           for label, color in zip(labels, colors)]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.088, .910), ncol=3,
           frameon=False, handlelength=2.0, columnspacing=1.8, fontsize=9.3)
type_handles = [Line2D([0], [0], color="#444444", lw=1.8, ls="--", label="ESS of weights w (without reward)"),
                Line2D([0], [0], color="#444444", lw=1.8, label="ESS of weights wᴿ (with reward)")]
fig.legend(handles=type_handles, loc="upper left", bbox_to_anchor=(.088, .868), ncol=2,
           frameon=False, handlelength=2.5, columnspacing=2.5)

for rows, color in zip(series, colors):
    x = [row["step_index"] + 1 for row in rows]
    axes[0].plot(x, [row["ess"] for row in rows], color=color, lw=1.3, ls="--", alpha=.8)
    axes[0].plot(x, [row["ess_reward"] for row in rows], color=color, lw=1.8)
    axes[1].plot(x, [100 * row["relative_noise_correction"] for row in rows],
                 color=color, lw=1.7, marker=".", markersize=3)

axes[0].set_title("A. The 20-step bank concentrates almost all weight on one candidate", loc="left", pad=10)
axes[0].set_ylabel("Effective sample size (ESS)")
axes[0].set_ylim(.85, 4.6)
axes[0].set_yticks([1, 2, 3, 4])
axes[0].axhline(1, color="#7B8794", lw=.8, alpha=.6)
axes[0].text(.99, .94, "Enlarged scale; theoretical maximum = bank size", transform=axes[0].transAxes,
             va="top", ha="right", color="#56616D", fontsize=9)
axes[0].annotate("8 × 20: ESSᴿ < 1.01 at 49/50 steps", xy=(35, 1.0), xytext=(27, 2.4),
                 fontsize=9, color=colors[2], arrowprops={"arrowstyle": "->", "color": colors[2], "lw": .9})

axes[1].set_title("B. Correction actually applied to CFG noise", loc="left", pad=10)
axes[1].set_ylabel("100 × ‖Δε‖₂ / ‖εCFG‖₂  (%)")
axes[1].set_xlabel("DDIM generation step (t decreases from 981 to 1)")
axes[1].set_yscale("symlog", linthresh=.001, linscale=.65)
axes[1].set_ylim(-.00015, 35)
axes[1].yaxis.set_major_locator(FixedLocator([0, .001, .01, .1, 1, 10]))
axes[1].yaxis.set_major_formatter(FixedFormatter(["0", "0.001", "0.01", "0.1", "1", "10"]))
peak = max(series[1], key=lambda row: row["relative_noise_correction"])
axes[1].annotate("16 × 5: 12.0 % at step 30 (t = 401)",
                 xy=(peak["step_index"] + 1, 100 * peak["relative_noise_correction"]),
                 xytext=(32, 17), fontsize=9, color=colors[1],
                 arrowprops={"arrowstyle": "->", "color": colors[1], "lw": 1})
axes[1].set_xlim(1, 50)
axes[1].set_xticks([1, 10, 20, 30, 40, 50])
for axis in axes:
    axis.grid(axis="both", color="#DCE1E6", lw=.7, alpha=.75)
    axis.set_axisbelow(True)

zero_counts = [sum(row["relative_noise_correction"] == 0 for row in rows) for rows in series]
fig.text(.095, .079, "Panel B: linear scale up to 0.001 %, logarithmic above; zero corrections are retained.",
         ha="left", fontsize=9, color="#454D56")
fig.text(.095, .052, f"Zero corrections: {zero_counts[0]}/50 (8 × 5), {zero_counts[1]}/50 (16 × 5), {zero_counts[2]}/50 (8 × 20). Each curve follows its own trajectory.",
         ha="left", fontsize=9, color="#454D56")
fig.text(.095, .025, "Source: JSON diagnostics from run lidar_ablations_20261011_153514_701207_UTC. Descriptive result; robustness not established.",
         ha="left", fontsize=8.1, color="#66717E")
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(DESTINATION, dpi=180, facecolor="white")
plt.close(fig)
print(DESTINATION)
