"""Contrôle CPU de l'image d'évaluation, sans modèle ni téléchargement de poids."""

import importlib.metadata
import json


def main():
    # HPSv2 importe turtle ; tkinter est nécessaire même sans interface graphique.
    import tkinter
    import torch
    import torchvision
    import clip
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    import evaluate
    from run_experiment import completed_result

    # Versions de la base Docker ; les variantes locales +cpu/+cu126 sont permises.
    assert torch.__version__.split("+")[0] == "2.8.0", torch.__version__
    assert torchvision.__version__.split("+")[0] == "0.23.0", torchvision.__version__
    evaluate.require_source_revision("clip", evaluate.CLIP_REVISION)
    evaluate.require_source_revision("hpsv2", evaluate.HPS_CODE_REVISION)
    assert callable(create_model_and_transforms) and callable(completed_result)

    # Ces tokenizers utilisent le vocabulaire livré avec les sources, sans poids.
    assert tuple(clip.tokenize(["a red flower"]).shape) == (1, 77)
    assert tuple(get_tokenizer("ViT-H-14")(["a red flower"]).shape) == (1, 77)
    assert not torch.cuda.is_initialized(), "Ce contrôle doit rester sur CPU."
    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "diffusers", "transformers", "clip", "hpsv2")}
    print(json.dumps({"imports": "ok", "source_revisions": "ok", "versions": versions,
                      "cuda_initialized": False, "model_weights_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
