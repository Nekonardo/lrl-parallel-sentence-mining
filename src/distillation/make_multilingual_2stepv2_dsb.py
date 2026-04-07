import logging
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import Dataset

from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import (
    MSEEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
)
from sentence_transformers.losses import MSELoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Teacher: strong multilingual model, we use it only on German (anchor language)
teacher_model_name = "sentence-transformers/LaBSE"

student_max_seq_length = 128

# You can tune these depending on your GPU
train_batch_size = 64 * 4          # per-device train batch size
inference_batch_size = 64 * 4      # for teacher encoding & evaluation

max_sentences_dsb_de = None        # use all dsb–de (usually small)

num_train_epochs_stage2 = 1        # Stage 2 (dsb–de, fine-tuning)

num_evaluation_steps = 5000

# Local data paths
_SCRIPT_DIR = Path(__file__).resolve().parent
data_dir = _SCRIPT_DIR.parent.parent / "data" / "distillation"

# Stage 1 model reused from a previous run (cs–de distillation already done).
# Override via environment variable STAGE1_MODEL_PATH, or edit this default.
import os
stage1_model_path = Path(
    os.environ.get("STAGE1_MODEL_PATH", "")
    or _SCRIPT_DIR.parent.parent
       / "output" / "distillation" / "model"
       / "make-multilingual-cs-de-hsb-two-stage-500k2026-03-17_05-16-05"
       / "stage1_cs-de" / "final"
)

dsb_german_data = {
    "source_file": data_dir / "MT" / "train.de-dsb.dsb",  # Lower Sorbian
    "target_file": data_dir / "MT" / "train.de-dsb.de",   # German
    "source_lang": "dsb",
    "target_lang": "de",
    "max_sentences": max_sentences_dsb_de,
}

_OUTPUT_BASE = _SCRIPT_DIR.parent.parent / "output" / "distillation" / "model"
run_root = (
    _OUTPUT_BASE / ("make-multilingual-cs-de-dsb-two-stage-500k"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
)

# ============================================================================
# Data loading
# ============================================================================

def load_parallel_data(source_file: Path, target_file: Path, max_sentences: int | None = None) -> Dataset:
    """
    Load parallel sentences from two text files (one sentence per line).

    Assumption:
      - source_file: other language (cs or hsb)
      - target_file: German (de)

    We map:
      - 'english'      -> German sentences (anchor, teacher will encode this)
      - 'non_english'  -> other language sentences (cs or hsb)
    """
    logger.info(f"Loading parallel data from {source_file} (other) and {target_file} (German)")

    with open(source_file, "r", encoding="utf-8") as f_src, open(
        target_file, "r", encoding="utf-8"
    ) as f_tgt:
        other_sentences = [line.strip() for line in f_src if line.strip()]
        german_sentences = [line.strip() for line in f_tgt if line.strip()]

    min_len = min(len(other_sentences), len(german_sentences))
    other_sentences = other_sentences[:min_len]
    german_sentences = german_sentences[:min_len]

    if max_sentences is not None and min_len > max_sentences:
        other_sentences = other_sentences[:max_sentences]
        german_sentences = german_sentences[:max_sentences]

    logger.info(f"Loaded {len(german_sentences)} parallel sentence pairs")

    data = {
        "english": german_sentences,      # anchor: German
        "non_english": other_sentences,   # Czech or Upper Sorbian
    }
    return Dataset.from_dict(data)


def split_train_eval(full_dataset: Dataset, name: str):
    eval_size = min(1000, int(len(full_dataset) * 0.1))
    split_dataset = full_dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    logger.info(f"{name}: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    return train_dataset, eval_dataset

# ============================================================================
# Prepare data
# ============================================================================

logger.info("=" * 80)
logger.info("Loading local parallel data")
logger.info("=" * 80)

dsb_de_full = load_parallel_data(
    dsb_german_data["source_file"],
    dsb_german_data["target_file"],
    max_sentences=dsb_german_data["max_sentences"],
)

dsb_de_train, dsb_de_eval = split_train_eval(dsb_de_full, "dsb-de")

# ============================================================================
# Initialize models
# ============================================================================

logger.info("=" * 80)
logger.info("Initializing models")
logger.info("=" * 80)

teacher_model = SentenceTransformer(teacher_model_name)
logger.info(f"Teacher model: {teacher_model}")

# ============================================================================
# Helper: prepare dataset with teacher embeddings (labels)
# ============================================================================

logger.info("=" * 80)
logger.info("Defining dataset preparation function (teacher embeddings)")
logger.info("=" * 80)

def prepare_dataset(batch):
    """
    Add teacher embeddings as 'label'.

    - batch['english']      : German sentences (anchor language)
    - batch['non_english']  : Other language sentences (dsb)
    - label                 : LaBSE(German)
    """
    return {
        "english": batch["english"],
        "non_english": batch["non_english"],
        "label": teacher_model.encode(
            batch["english"],
            batch_size=inference_batch_size,
            show_progress_bar=False,
        ),
    }

# ============================================================================
# Stage 1: reuse existing cs–de model (skip training)
# ============================================================================

logger.info("=" * 80)
logger.info(f"Stage 1: Loading pre-trained cs–de model from {stage1_model_path}")
logger.info("=" * 80)

# ============================================================================
# Stage 2: De–dsb (fine-tuning)
# ============================================================================

logger.info("=" * 80)
logger.info("Stage 2: Fine-tuning on De–dsb with German as anchor")
logger.info("=" * 80)

student_model = SentenceTransformer(str(stage1_model_path))
student_model.max_seq_length = student_max_seq_length
logger.info(f"Student model (Stage1 init for Stage2): {student_model}")

train_loss_stage2 = MSELoss(model=student_model)

dsb_de_train_columns = dsb_de_train.column_names
dsb_de_eval_columns = dsb_de_eval.column_names

dsb_de_train_with_label = dsb_de_train.map(
    prepare_dataset,
    batched=True,
    batch_size=30_000,
    remove_columns=dsb_de_train_columns,
)
dsb_de_eval_with_label = dsb_de_eval.map(
    prepare_dataset,
    batched=True,
    batch_size=30_000,
    remove_columns=dsb_de_eval_columns,
)

logger.info(f"Stage2 dsb-de train dataset columns: {dsb_de_train_with_label.column_names}")
logger.info(f"Stage2 dsb-de eval dataset columns: {dsb_de_eval_with_label.column_names}")

evaluators_stage2 = []

dev_mse_dsb = MSEEvaluator(
    source_sentences=dsb_de_eval["english"],      # German
    target_sentences=dsb_de_eval["non_english"],  # Lower Sorbian
    name="dsb-de",
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
)
evaluators_stage2.append(dev_mse_dsb)

dev_trans_dsb = TranslationEvaluator(
    source_sentences=dsb_de_eval["english"],      # German
    target_sentences=dsb_de_eval["non_english"],  # Lower Sorbian
    name="dsb-de",
    batch_size=inference_batch_size,
)
evaluators_stage2.append(dev_trans_dsb)

evaluator_stage2 = SequentialEvaluator(
    evaluators_stage2,
    main_score_function=lambda scores: float(np.mean(scores)),
)

args_stage2 = SentenceTransformerTrainingArguments(
    output_dir=str(Path(run_root) / "stage2_de-dsb"),
    num_train_epochs=num_train_epochs_stage2,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,
    bf16=False,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=num_evaluation_steps,
    save_strategy="steps",
    save_steps=num_evaluation_steps,
    save_total_limit=2,
    logging_steps=100,
    run_name="multilingual-stage2-de-dsb",
)

trainer_stage2 = SentenceTransformerTrainer(
    model=student_model,
    args=args_stage2,
    train_dataset=dsb_de_train_with_label,
    eval_dataset=dsb_de_eval_with_label,
    loss=train_loss_stage2,
    evaluator=evaluator_stage2,
)

logger.info("***** Start Stage 2 training *****")
trainer_stage2.train()

final_output_dir = Path(run_root) / "stage2_de-dsb" / "final"
final_output_dir.parent.mkdir(parents=True, exist_ok=True)
student_model.save(str(final_output_dir))
logger.info(f"Final model (after Stage 2) saved to {final_output_dir}")

logger.info("=" * 80)
logger.info("Training completed!")
logger.info("=" * 80)
