"""Comparer les diagnostics LiDAR archivés, sans exécuter le modèle."""
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
parser.add_argument("--input", type=Path, required=True, help="Archive ZIP des ablations T4 historiques.")
parser.add_argument("--output", type=Path, required=True, help="Fichier PNG à écrire.")
args = parser.parse_args()
ARCHIVE = args.input
DESTINATION = args.output
paths = ["reference_8x5/LiDAR_1_steps.json", "bank16_steps5/steps.json", "bank8_steps20/steps.json"]
labels = ["8 candidats × 5 pas (référence)", "16 candidats × 5 pas", "8 candidats × 20 pas"]
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
fig.suptitle("LiDAR : deux ablations de la banque à β constant", x=.095, y=.972,
             ha="left", fontsize=16, fontweight="bold")
fig.text(.095, .928, "Pilote T4 · 1 prompt, 1 seed · β = 66,3703 · génération finale à 50 pas DDIM",
         ha="left", fontsize=10, color="#454D56")
handles = [Line2D([0], [0], color=color, lw=2.3, label=label)
           for label, color in zip(labels, colors)]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.088, .910), ncol=3,
           frameon=False, handlelength=2.0, columnspacing=1.8, fontsize=9.3)
type_handles = [Line2D([0], [0], color="#444444", lw=1.8, ls="--", label="ESS des poids w (sans récompense)"),
                Line2D([0], [0], color="#444444", lw=1.8, label="ESS des poids wᴿ (avec récompense)")]
fig.legend(handles=type_handles, loc="upper left", bbox_to_anchor=(.088, .868), ncol=2,
           frameon=False, handlelength=2.5, columnspacing=2.5)

for rows, color in zip(series, colors):
    x = [row["step_index"] + 1 for row in rows]
    axes[0].plot(x, [row["ess"] for row in rows], color=color, lw=1.3, ls="--", alpha=.8)
    axes[0].plot(x, [row["ess_reward"] for row in rows], color=color, lw=1.8)
    axes[1].plot(x, [100 * row["relative_noise_correction"] for row in rows],
                 color=color, lw=1.7, marker=".", markersize=3)

axes[0].set_title("A. La banque à 20 pas concentre presque tout le poids sur un candidat", loc="left", pad=10)
axes[0].set_ylabel("Taille effective (ESS)")
axes[0].set_ylim(.85, 4.6)
axes[0].set_yticks([1, 2, 3, 4])
axes[0].axhline(1, color="#7B8794", lw=.8, alpha=.6)
axes[0].text(.99, .94, "Échelle agrandie ; maximum théorique = taille de la banque", transform=axes[0].transAxes,
             va="top", ha="right", color="#56616D", fontsize=9)
axes[0].annotate("8 × 20 : ESSᴿ < 1,01 sur 49/50 pas", xy=(35, 1.0), xytext=(27, 2.4),
                 fontsize=9, color=colors[2], arrowprops={"arrowstyle": "->", "color": colors[2], "lw": .9})

axes[1].set_title("B. Correction réellement appliquée au bruit CFG", loc="left", pad=10)
axes[1].set_ylabel("100 × ‖Δε‖₂ / ‖εCFG‖₂  (%)")
axes[1].set_xlabel("Pas de génération DDIM (t décroît de 981 à 1)")
axes[1].set_yscale("symlog", linthresh=.001, linscale=.65)
axes[1].set_ylim(-.00015, 35)
axes[1].yaxis.set_major_locator(FixedLocator([0, .001, .01, .1, 1, 10]))
axes[1].yaxis.set_major_formatter(FixedFormatter(["0", "0,001", "0,01", "0,1", "1", "10"]))
peak = max(series[1], key=lambda row: row["relative_noise_correction"])
axes[1].annotate("16 × 5 : 12,0 % au pas 30 (t = 401)",
                 xy=(peak["step_index"] + 1, 100 * peak["relative_noise_correction"]),
                 xytext=(32, 17), fontsize=9, color=colors[1],
                 arrowprops={"arrowstyle": "->", "color": colors[1], "lw": 1})
axes[1].set_xlim(1, 50)
axes[1].set_xticks([1, 10, 20, 30, 40, 50])
for axis in axes:
    axis.grid(axis="both", color="#DCE1E6", lw=.7, alpha=.75)
    axis.set_axisbelow(True)

zero_counts = [sum(row["relative_noise_correction"] == 0 for row in rows) for rows in series]
fig.text(.095, .079, "Panneau B : échelle linéaire jusqu’à 0,001 %, logarithmique au-delà ; les corrections nulles sont conservées.",
         ha="left", fontsize=9, color="#454D56")
fig.text(.095, .052, f"Corrections nulles : {zero_counts[0]}/50 (8 × 5), {zero_counts[1]}/50 (16 × 5), {zero_counts[2]}/50 (8 × 20). Chaque courbe suit sa propre trajectoire.",
         ha="left", fontsize=9, color="#454D56")
fig.text(.095, .025, "Source : diagnostics JSON du run lidar_ablations_20261011_153514_701207_UTC. Résultat descriptif, sans conclusion de robustesse.",
         ha="left", fontsize=8.1, color="#66717E")
DESTINATION.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(DESTINATION, dpi=180, facecolor="white")
plt.close(fig)
print(DESTINATION)
