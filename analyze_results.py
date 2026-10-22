"""Analyse descriptive du CSV de evaluate.py, sans modèle ni génération.

Exemple : python analyze_results.py --input runs/evaluation.csv --output runs/analysis
Les écarts-types décrivent les observations ; ils ne sont pas des erreurs standard.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


NUMERIC_FIELDS = (
    "beta", "reward_png", "reward_tensor", "clip_score", "hpsv2", "seconds",
    "peak_allocated_mib", "bank_generation_seconds", "bank_scoring_seconds",
)
GROUP_FIELDS = ("split", "reward_name", "reward_parameters_json", "method", "beta")


def read_results(path):
    """Lit le CSV ; une métrique manquante reste None, jamais zéro."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"config_id", "result_id", "prompt", "seed", "method", "reward_name", "reward_png"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Colonnes manquantes : {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for line, source in enumerate(reader, start=2):
            row = dict(source)
            row["seed"] = int(row["seed"])
            row["method"] = row["method"].lower()
            if row["method"] not in ("base", "cov", "full", "lidar"):
                raise ValueError(f"Méthode inconnue à la ligne {line}")
            if not row["result_id"] or not row["prompt"] or not row["reward_name"]:
                raise ValueError(f"Identité incomplète à la ligne {line}")
            for field in NUMERIC_FIELDS:
                value = row.get(field)
                row[field] = float(value) if value not in (None, "") else None
                if row[field] is not None and not math.isfinite(row[field]):
                    raise ValueError(f"Valeur non finie : ligne {line}, {field}")
            for field in ("seconds", "peak_allocated_mib", "bank_generation_seconds", "bank_scoring_seconds"):
                if row[field] is not None and row[field] < 0:
                    raise ValueError(f"Coût négatif : ligne {line}, {field}")
            row["split"] = row.get("split", "")
            parameters = json.loads(row.get("reward_parameters_json") or "{}")
            row["reward_parameters_json"] = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
            rows.append(row)
    if not rows:
        raise ValueError("Le CSV ne contient aucun résultat.")
    return rows


def summarize(rows):
    """Apparie à BASE, agrège les métriques et alloue une banque à ses utilisateurs."""
    config_ids = {row.get("config_id") for row in rows}
    if len(config_ids) != 1 or not next(iter(config_ids)):
        raise ValueError("Analyser une seule campagne config_id à la fois, avec une identité non vide.")
    # L'indice dépend du fichier de prompts : on apparie sur le texte exact.
    pair_fields = ("split", "prompt", "seed", "reward_name", "reward_parameters_json")
    baselines, seen, unique_generations = {}, set(), {}
    bank_generation, bank_scoring = {}, {}
    bank_users, score_users = defaultdict(set), defaultdict(set)
    for row in rows:
        identity = (row["result_id"], row["reward_name"], row["reward_parameters_json"])
        if identity in seen:
            raise ValueError(f"Résultat dupliqué dans le CSV : {identity}")
        seen.add(identity)
        result_id = row["result_id"]
        seconds = row["seconds"]
        if result_id in unique_generations and unique_generations[result_id] != seconds:
            raise ValueError(f"Temps contradictoires pour {result_id}")
        unique_generations[result_id] = seconds
        if row["method"] == "base":
            key = tuple(row[field] for field in pair_fields)
            if key in baselines:
                raise ValueError(f"Deux BASE possibles pour le même couple prompt/seed : {key}")
            baselines[key] = row
        elif row["beta"] is None:
            raise ValueError(f"Beta absent pour {result_id}")
        if row["method"] != "lidar":
            continue
        bank_id = row.get("bank_id")
        score_key = row.get("bank_score_key")
        if not bank_id or not score_key:
            raise ValueError(f"Identité de banque/scoring absente pour {result_id}")
        bank_users[bank_id].add(result_id)
        scoring_id = (bank_id, score_key)
        score_users[scoring_id].add(result_id)
        for cache, key, field in (
            (bank_generation, bank_id, "bank_generation_seconds"),
            (bank_scoring, scoring_id, "bank_scoring_seconds"),
        ):
            value = row[field]
            if key in cache and cache[key] != value:
                raise ValueError(f"Coûts de banque contradictoires pour {key}, {field}")
            cache[key] = value

    paired_rows, groups = [], defaultdict(list)
    missing_pairs = []
    for source in rows:
        row = dict(source)
        baseline = baselines.get(tuple(row[field] for field in pair_fields))
        row["base_result_id"] = baseline["result_id"] if baseline else ""
        row["base_reward_png"] = baseline["reward_png"] if baseline else None
        row["reward_gain"] = None
        if row["reward_png"] is not None and row["base_reward_png"] is not None:
            row["reward_gain"] = row["reward_png"] - row["base_reward_png"]
        elif row["method"] != "base":
            missing_pairs.append({"result_id": row["result_id"], "reason": "missing BASE or reward"})
        row["bank_generation_users"] = 0
        row["bank_scoring_users"] = 0
        row["bank_share_seconds"] = 0.0
        row["standalone_seconds"] = row["seconds"]
        row["amortized_seconds"] = row["seconds"]
        if row["method"] == "lidar":
            bank_id = row["bank_id"]
            scoring_id = (bank_id, row["bank_score_key"])
            row["bank_generation_users"] = len(bank_users[bank_id])
            row["bank_scoring_users"] = len(score_users[scoring_id])
            generation, scoring = bank_generation[bank_id], bank_scoring[scoring_id]
            if generation is None or scoring is None:
                row["bank_share_seconds"] = None
                row["standalone_seconds"] = None
                row["amortized_seconds"] = None
            else:
                row["bank_share_seconds"] = generation / len(bank_users[bank_id]) + scoring / len(score_users[scoring_id])
                if row["seconds"] is not None:
                    row["standalone_seconds"] = row["seconds"] + generation + scoring
                    row["amortized_seconds"] = row["seconds"] + row["bank_share_seconds"]
        paired_rows.append(row)
        groups[tuple(row.get(field) for field in GROUP_FIELDS)].append(row)

    summaries = []
    fields = ("reward_png", "reward_gain", "clip_score", "hpsv2", "seconds", "peak_allocated_mib", "bank_share_seconds", "standalone_seconds", "amortized_seconds")
    for key, observations in groups.items():
        summary = dict(zip(GROUP_FIELDS, key))
        summary["n_images"] = len(observations)
        summary["n_prompts"] = len({row["prompt"] for row in observations})
        for field in fields:
            values = [row[field] for row in observations if row[field] is not None]
            summary[f"{field}_n"] = len(values)
            summary[f"{field}_mean"] = mean(values) if values else None
            summary[f"{field}_std"] = stdev(values) if len(values) > 1 else None
        gains = [row["reward_gain"] for row in observations if row["reward_gain"] is not None]
        summary["n_unpaired"] = len(observations) - len(gains)
        summary["fraction_improved"] = sum(gain > 0 for gain in gains) / len(gains) if gains else None
        summary["fraction_unchanged"] = sum(gain == 0 for gain in gains) / len(gains) if gains else None
        prompt_gains = defaultdict(list)
        for row in observations:
            if row["reward_gain"] is not None:
                prompt_gains[row["prompt"]].append(row["reward_gain"])
        prompt_means = [mean(values) for values in prompt_gains.values()]
        summary["prompt_mean_gain_n"] = len(prompt_means)
        summary["prompt_mean_gain_mean"] = mean(prompt_means) if prompt_means else None
        summary["prompt_mean_gain_std"] = stdev(prompt_means) if len(prompt_means) > 1 else None
        peaks = [row["peak_allocated_mib"] for row in observations if row["peak_allocated_mib"] is not None]
        summary["peak_allocated_mib_max"] = max(peaks) if peaks else None
        summaries.append(summary)

    def complete_sum(values):
        values = list(values)
        return sum(values) if all(value is not None for value in values) else None

    totals = {
        "config_id": next(iter(config_ids)),
        "n_csv_rows": len(rows), "n_unique_generations": len(unique_generations),
        "generation_seconds_unique": complete_sum(unique_generations.values()),
        "bank_generation_seconds_unique": complete_sum(bank_generation.values()),
        "bank_scoring_seconds_unique": complete_sum(bank_scoring.values()),
        "n_banks": len(bank_generation), "n_bank_reward_scores": len(bank_scoring),
        "missing_pairs": missing_pairs,
        "scope": "Completed rows in this CSV only; this file alone cannot determine failed or unscheduled tasks.",
        "amortization": "Generation bank cost / distinct LiDAR result_ids sharing bank_id, plus scoring cost / distinct result_ids sharing (bank_id, bank_score_key), within this CSV. Changing this subset changes the denominator.",
        "uncertainty": "Sample standard deviations (ddof=1); undefined for n<2. No independence assumption, confidence interval or significance test. Prompt-level gains average available seeds per prompt.",
        "reward_convention": "Paired rewards are recomputed from PNG. reward_tensor is retained separately; no mixing of continuous and quantized-image scores.",
    }
    return summaries, paired_rows, totals


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(summaries, output):
    """Une figure par récompense/configuration ; barres = moyenne, erreur = écart-type."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in summaries:
        groups[(row["split"], row["reward_name"], row["reward_parameters_json"])].append(row)
    for index, (key, rows) in enumerate(groups.items()):
        labels = [f"{row['method'].upper()}\nβ={row['beta']:g}" if row["method"] != "base" else "BASE" for row in rows]
        figure, axes = plt.subplots(2, 3, figsize=(max(12, 1.35 * len(rows)), 8))
        metrics = (
            ("reward_png", "Reward from PNG"), ("reward_gain", "Paired reward gain vs BASE"),
            ("clip_score", "CLIPScore"), ("hpsv2", "HPSv2"),
            ("amortized_seconds", "Seconds / image, bank amortized"),
            ("peak_allocated_mib", "Peak allocated GPU memory (MiB)"),
        )
        for axis, (metric, title) in zip(axes.flat, metrics):
            for position, row in enumerate(rows):
                value, deviation = row[f"{metric}_mean"], row[f"{metric}_std"]
                if value is None:
                    axis.text(position, 0.5, "missing", rotation=90, va="center", ha="center", transform=axis.get_xaxis_transform())
                    continue
                axis.bar(position, value, color="#4176a6")
                if deviation is not None:
                    axis.errorbar(position, value, yerr=deviation, fmt="none", capsize=3, color="black")
            axis.axhline(0, color="gray", linewidth=0.6)
            axis.set_xticks(range(len(rows)), labels, fontsize=8)
            axis.set_xlim(-0.6, len(rows) - 0.4)
            axis.set_title(title, fontsize=11)
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(f"{key[0]} · {key[1]} · {key[2]}\nDescriptive means ± sample SD; see counts and missing pairs in summary.csv", fontsize=11)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(Path(output) / f"comparison_{index:02d}_{key[1]}.png", dpi=180)
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV de evaluate.py, une campagne à la fois")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true", help="Agrégation stdlib seule, sans matplotlib")
    args = parser.parse_args(argv)
    summaries, paired_rows, totals = summarize(read_results(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "paired_results.csv", paired_rows)
    (args.output / "analysis_metadata.json").write_text(json.dumps(totals, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Descriptive analysis", "", totals["scope"], "", totals["uncertainty"], "", totals["amortization"], "",
             "A strictly positive PNG reward difference counts as improved. BASE rows have zero gain by construction.", "",
             "Inspect the campaign failure log and planned matrix before interpreting completion rates or robustness.", "",
             "| Reward | Method | Beta | Images | Paired | Reward mean | Gain mean | Improved |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        values = [row["reward_name"], row["method"], row["beta"], row["n_images"], row["reward_gain_n"], row["reward_png_mean"], row["reward_gain_mean"], row["fraction_improved"]]
        lines.append("| " + " | ".join("—" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value) for value in values) + " |")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.no_plots:
        make_figures(summaries, args.output)
    print(json.dumps({"output": str(args.output), "groups": len(summaries), "missing_pairs": len(totals["missing_pairs"])}, indent=2))
    return 2 if totals["missing_pairs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
