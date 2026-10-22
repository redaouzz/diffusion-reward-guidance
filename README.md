# Diffusion Reward Guidance

A personal research project on reward-guided Stable Diffusion 1.5 sampling with DDIM.
The repository contains the T4 development experiments and the H100 luminance benchmark accompanying the [French report](docs/report.pdf) and [English report](docs/report_en.pdf).

- **BASE**: DDIM with classifier-free guidance, without reward guidance (`baseline.py`).
- **COV**: reward gradient with an isotropic covariance approximation (`guidance_cov.py`).
- **FULL**: reward gradient through the Tweedie prediction and both CFG branches (`guidance_full.py`).
- **LiDAR**: reward-weighted independent lookahead samples (`guidance_lidar.py`).

## Environment

Use Linux/amd64, Docker with BuildKit, and the NVIDIA Container Toolkit for GPU runs.
An NVIDIA GPU is required for the full experiments; the final runs used an H100 80 GB.
From the repository root, build the two independent, pinned environments:

```bash
docker build --platform linux/amd64 --target generation -t diffusion-reward-guidance:generation .
docker build --platform linux/amd64 --target evaluation -t diffusion-reward-guidance:evaluation .
docker volume create guidance-data
```

Both images use `/data/cache` for weights downloaded at runtime. Keep `/data` on persistent storage.
For Northflank, mount the existing volume there and use the `python ...` portion of each command below in the indicated image.

## Reproduce the experiments

DEV calibration, using the candidate grid from the report:

```bash
docker run --rm --gpus all -v guidance-data:/data diffusion-reward-guidance:generation python calibrate_beta.py \
  --prompts data/prompts_50_dev.txt --exclude-prompts data/prompts_50_heldout.txt \
  --limit 5 --betas 23.09 66.37033847314085 115.46 --output /data/results/beta_calibration_5p3b --device cuda
```

Final benchmark (the historical `10p` filename contains the actual eight-prompt configuration):

```bash
docker run --rm --gpus all -v guidance-data:/data diffusion-reward-guidance:generation python run_experiment.py \
  --config configs/benchmark_luminance_10p.json --output /data/results/benchmark_luminance_8p --device cuda
```

Evaluate saved PNGs, then produce tables and figures:

```bash
docker run --rm --gpus all -v guidance-data:/data diffusion-reward-guidance:evaluation python evaluate.py \
  --input /data/results/benchmark_luminance_8p --metrics reward clip hps --device cuda --cache-dir /data/cache
docker run --rm -v guidance-data:/data diffusion-reward-guidance:generation python analyze_results.py \
  --input /data/results/benchmark_luminance_8p/evaluation.csv --output /data/results/benchmark_luminance_8p/analysis
```

Repeat the same generation command and output path to resume; completed artefacts are checked before reuse.
To check the benchmark matrix without weights or a GPU, replace `--device cuda` with `--device cpu --dry-run` and omit `--gpus all`.
Run the CPU tests with `docker run --rm diffusion-reward-guidance:generation python -m unittest discover -v`.

## Files and outputs

- Root Python files: samplers, rewards, campaign scripts and numerical/reproducibility tests.
- `configs/`, `data/`: experiment configurations, prompt splits and the small T4 reference fixture.
- `notebooks/`, `validation/`, `analysis/`: T4 experiments, retained evidence and figure script.
- Each run stores its manifest, provenance, per-image PNG/metadata, diagnostics and LiDAR banks. Evaluation writes `evaluation.csv`; analysis writes tables and plots.

See the [French report](docs/report.pdf) and [English report](docs/report_en.pdf) for derivations, experimental choices, results and limitations.

LaTeX sources and figures are in `docs/latex/fr/` and `docs/latex/en/`; compile `main.tex` twice with pdfLaTeX.
The original benchmark CSV is in `data/benchmark_evaluation.csv`. Archived audit paths are normalized for this release; measurements and historical hashes are retained.
Rebuild the English LiDAR plot with `python analysis/plot_lidar_ablations_en.py --input data/lidar_ablations --output lidar_diagnostics_en.png`; use `plot_lidar_ablations.py` for French labels.
