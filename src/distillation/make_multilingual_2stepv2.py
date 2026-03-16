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

# Student: multilingual encoder to be distilled
# student_model_name = "xlm-roberta-base"
student_model_name = "cis-lmu/glot500-base"

student_max_seq_length = 128

# You can tune these depending on your GPU
train_batch_size = 64 * 4          # per-device train batch size
inference_batch_size = 64 * 4      # for teacher encoding & evaluation

max_sentences_cs_de = 500_000      # max number of cs–de parallel sentences
max_sentences_hsb_de = None        # use all hsb–de (usually small)

num_train_epochs_stage1 = 5        # Stage 1 (cs–de)
num_train_epochs_stage2 = 2        # Stage 2 (hsb–de, fine-tuning)

num_evaluation_steps = 5000

# Local data paths
data_dir = Path(__file__).parent / "data"

czech_german_data = {
    "source_file": data_dir / "Europarl.cs-de.cs",              # Czech
    "target_file": data_dir / "Europarl.cs-de.de",              # German
    "source_lang": "cs",
    "target_lang": "de",
    "max_sentences": max_sentences_cs_de,
}

# hsb_german_data = {
#     "source_file": data_dir / "raw_de-hsb" / "parallel_de-hsb_hsb.txt",  # Upper Sorbian
#     "target_file": data_dir / "raw_de-hsb" / "parallel_de-hsb_de.txt",   # German
#     "source_lang": "hsb",
#     "target_lang": "de",
#     "max_sentences": max_sentences_hsb_de,
# }

hsb_german_data = {
    "source_file": data_dir / "MT" / "train.de-hsb.hsb",  # Upper Sorbian
    "target_file": data_dir / "MT" / "train.de-hsb.de",   # German
    "source_lang": "hsb",
    "target_lang": "de",
    "max_sentences": max_sentences_hsb_de,
}

run_root = (
    f"output/make-multilingual-cs-de-hsb-two-stage-500k"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# ============================================================================
# Data loading
# ============================================================================

def load_parallel_data(source_file: Path, target_file: Path, max_sentences: int | None = None) -> Dataset:
    """
    Load parallel sentences from two text files (one sentence per line).

    约定:
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

cs_de_full = load_parallel_data(
    czech_german_data["source_file"],
    czech_german_data["target_file"],
    max_sentences=czech_german_data["max_sentences"],
)
hsb_de_full = load_parallel_data(
    hsb_german_data["source_file"],
    hsb_german_data["target_file"],
    max_sentences=hsb_german_data["max_sentences"],
)

cs_de_train, cs_de_eval = split_train_eval(cs_de_full, "cs-de")
hsb_de_train, hsb_de_eval = split_train_eval(hsb_de_full, "hsb-de")

# ============================================================================
# Initialize models
# ============================================================================

logger.info("=" * 80)
logger.info("Initializing models")
logger.info("=" * 80)

teacher_model = SentenceTransformer(teacher_model_name)
logger.info(f"Teacher model: {teacher_model}")

student_model = SentenceTransformer(student_model_name)
student_model.max_seq_length = student_max_seq_length
logger.info(f"Student model (initial): {student_model}")

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
    - batch['non_english']  : Other language sentences (cs or hsb)
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
# Stage 1: De–Cs
# ============================================================================

logger.info("=" * 80)
logger.info("Stage 1: Training on De–Cs with German as anchor")
logger.info("=" * 80)

cs_de_train_columns = cs_de_train.column_names
cs_de_eval_columns = cs_de_eval.column_names

cs_de_train_with_label = cs_de_train.map(
    prepare_dataset,
    batched=True,
    batch_size=30_000,
    remove_columns=cs_de_train_columns,
)
cs_de_eval_with_label = cs_de_eval.map(
    prepare_dataset,
    batched=True,
    batch_size=30_000,
    remove_columns=cs_de_eval_columns,
)

logger.info(f"Stage1 cs-de train dataset columns: {cs_de_train_with_label.column_names}")
logger.info(f"Stage1 cs-de eval dataset columns: {cs_de_eval_with_label.column_names}")

# Loss for Stage 1
train_loss_stage1 = MSELoss(model=student_model)

# Evaluators for Stage 1 (MSE + translation accuracy, de ↔ cs)
evaluators_stage1 = []

dev_mse_cs = MSEEvaluator(
    source_sentences=cs_de_eval["english"],      # German
    target_sentences=cs_de_eval["non_english"],  # Czech
    name="cs-de",
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
)
evaluators_stage1.append(dev_mse_cs)

dev_trans_cs = TranslationEvaluator(
    source_sentences=cs_de_eval["english"],      # German
    target_sentences=cs_de_eval["non_english"],  # Czech
    name="cs-de",
    batch_size=inference_batch_size,
)
evaluators_stage1.append(dev_trans_cs)

evaluator_stage1 = SequentialEvaluator(
    evaluators_stage1,
    main_score_function=lambda scores: float(np.mean(scores)),
)

args_stage1 = SentenceTransformerTrainingArguments(
    output_dir=str(Path(run_root) / "stage1_cs-de"),
    num_train_epochs=num_train_epochs_stage1,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,   # set False if your GPU does not support FP16
    bf16=False,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=num_evaluation_steps,
    save_strategy="steps",
    save_steps=num_evaluation_steps,
    save_total_limit=2,
    logging_steps=100,
    run_name="multilingual-stage1-cs-de",
)

trainer_stage1 = SentenceTransformerTrainer(
    model=student_model,
    args=args_stage1,
    train_dataset=cs_de_train_with_label,
    eval_dataset=cs_de_eval_with_label,
    loss=train_loss_stage1,
    evaluator=evaluator_stage1,
)

logger.info("***** Start Stage 1 training *****")
trainer_stage1.train()

stage1_output_dir = Path(run_root) / "stage1_cs-de" / "final"
stage1_output_dir.parent.mkdir(parents=True, exist_ok=True)
student_model.save(str(stage1_output_dir))
logger.info(f"Stage 1 model saved to {stage1_output_dir}")

# ============================================================================
# Stage 2: De–hsb (fine-tuning)
# ============================================================================

logger.info("=" * 80)
logger.info("Stage 2: Fine-tuning on De–hsb with German as anchor")
logger.info("=" * 80)

# Reload Stage1 model explicitly (safer & clear)
student_model = SentenceTransformer(str(stage1_output_dir))
student_model.max_seq_length = student_max_seq_length
logger.info(f"Student model (Stage1 init for Stage2): {student_model}")

train_loss_stage2 = MSELoss(model=student_model)

hsb_de_train_columns = hsb_de_train.column_names
hsb_de_eval_columns = hsb_de_eval.column_names

hsb_de_train_with_label = hsb_de_train.map(
    prepare_dataset,
    batched=True,
    batch_size=30_000,
    remove_columns=hsb_de_train_columns,
)
hsb_de_eval_with_label = hsb_de_eval.map(
    prepare_dataset,
    batched=True,
    batch_size=30_000,
    remove_columns=hsb_de_eval_columns,
)

logger.info(f"Stage2 hsb-de train dataset columns: {hsb_de_train_with_label.column_names}")
logger.info(f"Stage2 hsb-de eval dataset columns: {hsb_de_eval_with_label.column_names}")

evaluators_stage2 = []

dev_mse_hsb = MSEEvaluator(
    source_sentences=hsb_de_eval["english"],      # German
    target_sentences=hsb_de_eval["non_english"],  # hsb
    name="hsb-de",
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
)
evaluators_stage2.append(dev_mse_hsb)

dev_trans_hsb = TranslationEvaluator(
    source_sentences=hsb_de_eval["english"],      # German
    target_sentences=hsb_de_eval["non_english"],  # hsb
    name="hsb-de",
    batch_size=inference_batch_size,
)
evaluators_stage2.append(dev_trans_hsb)

evaluator_stage2 = SequentialEvaluator(
    evaluators_stage2,
    main_score_function=lambda scores: float(np.mean(scores)),
)

args_stage2 = SentenceTransformerTrainingArguments(
    output_dir=str(Path(run_root) / "stage2_de-hsb"),
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
    run_name="multilingual-stage2-de-hsb",
)

trainer_stage2 = SentenceTransformerTrainer(
    model=student_model,
    args=args_stage2,
    train_dataset=hsb_de_train_with_label,
    eval_dataset=hsb_de_eval_with_label,
    loss=train_loss_stage2,
    evaluator=evaluator_stage2,
)

logger.info("***** Start Stage 2 training *****")
trainer_stage2.train()

final_output_dir = Path(run_root) / "stage2_de-hsb" / "final"
final_output_dir.parent.mkdir(parents=True, exist_ok=True)
student_model.save(str(final_output_dir))
logger.info(f"Final model (after Stage 2) saved to {final_output_dir}")

# (Optional) Push to Hugging Face Hub
model_name = student_model_name if "/" not in student_model_name else student_model_name.split("/")[-1]
try:
    student_model.push_to_hub(f"{model_name}-multilingual-cs-de-hsb-two-stage")
except Exception:
    logging.error(
        "Error uploading model to the Hugging Face Hub:\n"
        f"{traceback.format_exc()}"
        "To upload it manually, run `huggingface-cli login`, then:\n"
        f"  model = SentenceTransformer({repr(str(final_output_dir))})\n"
        f"  model.push_to_hub('{model_name}-multilingual-cs-de-hsb-two-stage')"
    )

logger.info("=" * 80)
logger.info("Two-stage training completed!")
logger.info("=" * 80)
