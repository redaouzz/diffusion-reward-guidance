# Image PyTorch officielle linux/amd64 ; CUDA 12.6, PyTorch 2.8.0, cuDNN 9.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69 AS generation

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /app
COPY requirements.txt requirements-rewards.txt requirements-experiment.txt ./
RUN python -m pip install --no-cache-dir pip==25.2 setuptools==80.9.0 wheel==0.45.1
RUN python -m pip install --no-cache-dir -r requirements-experiment.txt
# Leurs setup.py utilisent encore pkg_resources ; garder les outils de build ci-dessus.
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation \
    "clip @ https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip" \
    "image-reward @ https://github.com/zai-org/ImageReward/archive/78f8051059b34fef8e0032ccb40561c23a2ddbbb.zip"
RUN python -m pip check
COPY . .
RUN python -m pip freeze > /opt/guidance-environment.txt

# Le volume /data doit être monté dans Northflank. ENV ne crée pas de persistance.
# Aucune génération au démarrage : une commande explicite est nécessaire.
CMD ["python", "run_experiment.py", "--help"]

# Cible indépendante : BuildKit n'a pas besoin de construire generation.
# Réutiliser PyTorch/CUDA déjà présents, sans hériter d'ImageReward/datasets.
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime@sha256:dab81780fd94483b67b4b5679cc0024939b08e48540d39476d284cb29002ed69 AS evaluation

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/cache/huggingface \
    TORCH_HOME=/data/cache/torch \
    XDG_CACHE_HOME=/data/cache \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PIP_CONSTRAINT=/opt/pytorch-constraints.txt

WORKDIR /app
# Interdire au résolveur de remplacer la pile PyTorch/CUDA de l'image de base.
RUN python -c "import importlib.metadata as m; from pathlib import Path; names = ('torch', 'torchvision', 'torchaudio', 'triton'); Path('/opt/pytorch-constraints.txt').write_text(''.join(d.metadata['Name'] + '==' + d.version + '\n' for d in m.distributions() if d.metadata['Name'].lower() in names or d.metadata['Name'].lower().startswith('nvidia-')))"
COPY requirements-evaluation.txt ./
RUN python -m pip install --no-cache-dir pip==25.2 setuptools==80.9.0 wheel==0.45.1
RUN python -m pip install --no-cache-dir -r requirements-evaluation.txt
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation \
    "clip @ https://github.com/openai/CLIP/archive/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip" \
    "hpsv2 @ https://github.com/tgxs002/HPSv2/archive/866735ecaae999fa714bd9edfa05aa2672669ee3.zip"
RUN python -m pip check
# tkinter, importé par HPSv2, charge libX11 même sans interface graphique.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libx11-6 \
    && rm -rf /var/lib/apt/lists/*
# evaluate importe le contrôle d'intégrité du runner, puis le scheduler DDIM.
# Garder ces sources inchangées ; aucun modèle n'est chargé par leurs imports.
COPY evaluate.py rewards.py run_experiment.py baseline.py guidance_cov.py guidance_full.py guidance_lidar.py check_evaluation_imports.py ./
RUN CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python check_evaluation_imports.py
RUN python -m pip freeze > /opt/guidance-environment.txt
CMD ["python", "evaluate.py", "--help"]
