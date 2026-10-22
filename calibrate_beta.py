"""Préparer une petite grille DEV de beta communs, puis appeler le runner existant."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def make_config(prompts, exclude_prompts, limit, betas, reward="luminance", parameters=None):
    """Configuration DEV uniquement ; aucun choix automatique de beta."""
    if not 1 <= limit <= 10:
        raise ValueError("Utiliser 1 à 10 prompts DEV ; 3 à 5 recommandés pour la calibration.")
    if len(betas) not in (2, 3) or len(set(betas)) != len(betas):
        raise ValueError("Donner deux ou trois beta candidats distincts.")
    if any(not math.isfinite(beta) or beta <= 0 for beta in betas):
        raise ValueError("Les beta candidats doivent être finis et strictement positifs.")
    prompts = Path(prompts).resolve()
    exclude_prompts = Path(exclude_prompts).resolve()
    if prompts == exclude_prompts:
        raise ValueError("Les listes DEV et held-out doivent être distinctes.")
    if not prompts.is_file() or not exclude_prompts.is_file():
        raise FileNotFoundError("Les deux listes de prompts doivent exister.")
    return {
        "prompts_file": str(prompts),
        "exclude_prompts_file": str(exclude_prompts),
        "split": "dev",
        "prompt_limit": limit,
        "seeds": [0],
        "methods": ["base", "cov", "full", "lidar"],
        "rewards": [{"name": reward, "parameters": parameters or {}, "betas": list(betas)}],
        "height": 512,
        "width": 512,
        "num_steps": 50,
        "guidance_scale": 7.5,
        "tau2": 1.0,
        "bank": {"size": 16, "num_steps": 5, "seed_start": 10000},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True, type=Path, help="Liste DEV, jamais held-out.")
    parser.add_argument("--exclude-prompts", required=True, type=Path, help="Liste held-out à exclure.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--betas", required=True, nargs="+", type=float)
    parser.add_argument("--reward", choices=("luminance", "symmetry", "spatial_mask", "image_reward"), default="luminance")
    parser.add_argument("--reward-parameters", default="{}", help='Objet JSON ; ex. \'{"target": 0.7}\'.')
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="Valider la matrice sans charger de modèle.")
    args = parser.parse_args()
    try:
        parameters = json.loads(args.reward_parameters)
        if not isinstance(parameters, dict):
            raise ValueError("--reward-parameters doit être un objet JSON.")
        config = make_config(args.prompts, args.exclude_prompts, args.limit, args.betas,
                             args.reward, parameters)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    # Les chemins absolus restent valides lorsque la configuration est sauvée dans le run.
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output.resolve() / "calibration_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        parser.error("Ce dossier contient une autre calibration. Choisir un nouveau --output.")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).with_name("run_experiment.py")),
               "--config", str(config_path), "--output", str(args.output.resolve()),
               "--device", args.device]
    if args.dry_run:
        command.append("--dry-run")
    # Aucun classement, aucune sélection, aucune consultation de scores held-out.
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
