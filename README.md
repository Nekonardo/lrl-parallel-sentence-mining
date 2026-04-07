# lrl-parallel-sentence-mining

Parallel sentence mining for low-resource languages using multilingual embeddings and knowledge distillation.

## Setup

```bash
git submodule update --init --recursive
bash scripts/prepare_data.sh
```

`prepare_data.sh` copies BUCC-style benchmark data, downloads OPUS-Europarl cs-de/de-pl, and copies Sorbian MT data into `data/distillation/`.

---

## Benchmark: Bilingual Sentence Mining

Evaluates multilingual embedding models on BUCC-style sentence mining across 4 low-resource language pairs: `chv-ru`, `dsb-de`, `hsb-de`, `oci-es`.

### Pipeline

1. **Embed** (`scripts/run_embeddings.py`): generate sentence vectors. Uses `src/benchmark/contextual_sentence_embeddings.py` for most models and `src/benchmark/contextual_sentence_embeddings_sonar.py` for SONAR.
2. **Similarity** (`scripts/run_similarity.py`): compute CSLS similarity scores.
3. **Filter** (`scripts/run_filtering.py`): sweep thresholds and compute F1 / transfer gap.

Results written to `output/embeddings/filtering_summary.csv`.

### Supported models

`glot500`, `xlmr`, `labse`, `laser2`, `mmbert`, `sonar` (+ proxy variants `sonar-pol`, `sonar-kaz`, `sonar-pol-kaz`)

---

## Distillation: Cross-lingual Knowledge Distillation

Fine-tunes `glot500-base` for low-resource Sorbian languages (hsb, dsb) via MSE distillation from LaBSE.

### Training

```bash
# Two-stage: cs-de pretraining → hsb-de fine-tuning (default)
python src/distillation/make_multilingual_2step.py --target-lang hsb

# Two-stage targeting Lower Sorbian
python src/distillation/make_multilingual_2step.py --target-lang dsb

# Single-stage: train directly on target language only
python src/distillation/make_multilingual_2step.py --target-lang hsb --mode single-stage

# Resume from existing Stage 1 model
python src/distillation/make_multilingual_2step.py --target-lang dsb --stage1-model-path output/distillation/model/stage1/
```

Output saved to `output/distillation/model/`.