# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

HALW = "High Accuracy Lightweight AI-generated text detection." A binary classifier that distinguishes AI-generated from human-written text using linguistic/stylometric features (not raw embeddings) fed into a small 1D CNN. The lightweight goal is structural: feature-based input keeps the model small, so changes that pull in a transformer / large embedding model contradict the project intent.

This is a **research project** — the user iterates over different data pipelines, models, and datasets. Optimize for swap-ability: keep sections/functions small and clearly named, mirror the structure of existing pipelines when adding new ones, and avoid clever abstractions that make a candidate hard to drop in or remove.

## Repository Layout

The entire project currently lives in a single Colab notebook: `place-holder.ipynb`. There is no Python package, no `requirements.txt`, no test suite, and no build system. Treat the notebook as the source of truth and edit it with `NotebookEdit` (not `Edit`) so cell structure is preserved.

## Runtime Environment

The notebook is written for **Google Colab**, not local execution. It depends on:

- `google.colab.drive` for persisting the extracted feature CSV to Drive at `/content/drive/MyDrive/shanegrami_ai_human_features_lg_50k.csv`. Cell 1 short-circuits the expensive feature-extraction step if that file already exists — preserve that branch when refactoring.
- `kagglehub.dataset_download("shanegerami/ai-vs-human-text")` for the source dataset (`AI_Human.csv`), balanced to 50k AI + 50k human (100k total).
- `spacy` with `en_core_web_lg` + the `textdescriptives/all` pipeline component to produce ~70 numeric features per text. The install cell ends with `os.kill(os.getpid(), 9)` to force a runtime restart so spaCy/numpy reload — this is intentional, not a bug.
- The dependency install pins `numpy<2.0` because `textdescriptives` is incompatible with numpy 2.x; the resulting Colab environment conflicts (cupy, jax, opencv, etc.) are expected and benign for this notebook's code paths.

## Data Pipeline (read before changing any cell)

1. **Feature extraction** (Section 02): spaCy + textdescriptives produces a wide numeric DataFrame. The `text` column is dropped; only numeric features + `label` are kept.
2. **Column-wise NaN handling**: columns with >50% missing are dropped (`thresh=0.5`). Per-row NaN handling is done *after* scaling (step 4), not here.
3. **Split + scale** (Section 03): stratified 80/10/10 train/val/test, then `StandardScaler.fit_transform` on train and `transform` on val/test. Inputs are reshaped to `(N, n_features, 1)` for `Conv1D`.
4. **Row-wise NaN drop** happens *after* scaling — scaler can produce NaNs from all-NaN columns/rows the column threshold didn't catch. The test set is masked the same way inside the training cell, right before `model.evaluate`. If you reorder cells, keep that masking adjacent to its consumer.
5. **Model** (Section 04): `Conv1D(128) → BatchNorm → Flatten → Dense(256/128/64) → Dense(1, sigmoid)`, Adam @ 1e-3, binary cross-entropy. Saved as `cnn_synthetic_natural_classifier.h5` in the Colab working directory (not Drive).

## Conventions

- Section headers use `# NN-Section Name` markdown cells (e.g., `02-Load Dataset`). Match this when adding new sections.
- The notebook prints sanity-check shapes and NaN counts at each major step; keep those when modifying the pipeline — they are how problems get caught in Colab.
- `EarlyStopping` is constructed but **not** passed to `model.fit` (only `reduce_lr` is). If you wire it in, verify the val curve still has room — current 50-epoch runs rely on `restore_best_weights` not being active.
