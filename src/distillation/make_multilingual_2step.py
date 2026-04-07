import argparse
import logging
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
# Argument parsing
# ============================================================================

parser = argparse.ArgumentParser(description="Multi-step Teacher-Student Distillation")

parser.add_argument(
    "--mode",
    choices=["two-stage", "single-stage"],
    default="two-stage",
    help=(
        "two-stage: Stage1 on cs-de then Stage2 on target; "
        "single-stage: skip Stage1, train directly on target language only"
    ),
)
parser.add_argument(
    "--target-lang",
    choices=["hsb", "dsb"],
    default="hsb",
    help="Target low-resource language for Stage 2 (hsb=Upper Sorbian, dsb=Lower Sorbian)",
)
parser.add_argument(
    "--stage1-max-sentences",
    type=int,
    default=500_000,
    help="Max cs-de sentences for Stage 1 (ablation). Use 0 for all available.",
)
parser.add_argument(
    "--stage1-epochs",
    type=int,
    default=5,
    help="Number of training epochs for Stage 1 (only used in two-stage mode)",
)
parser.add_argument(
    "--stage2-epochs",
    type=int,
    default=2,
    help="Number of training epochs for Stage 2 (ablation)",
)
parser.add_argument(
    "--stage1-model-path",
    type=str,
    default=None,
    help=(
        "Path to a pre-trained Stage 1 model. When set, Stage 1 training is skipped "
        "and this model is used as the starting point for Stage 2 "
        "(only relevant in two-stage mode)."
    ),
)
parser.add_argument(
    "--student-model",
    type=str,
    default="cis-lmu/glot500-base",
    help="HuggingFace model name or path to use as the student model (default: cis-lmu/glot500-base)",
)

args = parser.parse_args()

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

teacher_model_name = "sentence-transformers/LaBSE"
student_model_name = args.student_model
student_max_seq_length = 128

train_batch_size = 64 * 4
inference_batch_size = 64 * 4
num_evaluation_steps = 5000

stage1_max_sentences = args.stage1_max_sentences if args.stage1_max_sentences > 0 else None
stage1_epochs = args.stage1_epochs
stage2_epochs = args.stage2_epochs
target_lang = args.target_lang
mode = args.mode

_DEFAULT_STUDENT = "cis-lmu/glot500-base"
_student_tag = "" if student_model_name == _DEFAULT_STUDENT else "_" + student_model_name.split("/")[-1]

# ============================================================================
# Output directory — name encodes all key parameters
# ============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent
data_dir = _SCRIPT_DIR.parent.parent / "data" / "distillation"
_OUTPUT_BASE = _SCRIPT_DIR.parent.parent / "output" / "distillation" / "model"

if mode == "two-stage":
    s1_size_tag = f"cs{stage1_max_sentences // 1000}k" if stage1_max_sentences else "csAll"
    run_name = f"distil_two-stage_{s1_size_tag}_s1e{stage1_epochs}_s2e{stage2_epochs}_{target_lang}{_student_tag}"
else:
    run_name = f"distil_single-stage_s2e{stage2_epochs}_{target_lang}{_student_tag}"

run_root = _OUTPUT_BASE / (run_name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

logger.info("=" * 80)
logger.info(f"Experiment: {run_name}")
logger.info(f"Mode: {mode} | Target lang: {target_lang} | Output: {run_root}")
logger.info("=" * 80)

# ============================================================================
# Data paths
# ============================================================================

target_data_files = {
    "hsb": {
        "source_file": data_dir / "MT" / "train.de-hsb.hsb",
        "target_file": data_dir / "MT" / "train.de-hsb.de",
    },
    "dsb": {
        "source_file": data_dir / "MT" / "train.de-dsb.dsb",
        "target_file": data_dir / "MT" / "train.de-dsb.de",
    },
}

# ============================================================================
# Data loading
# ============================================================================

def load_parallel_data(source_file: Path, target_file: Path, max_sentences: int | None = None) -> Dataset:
    """
    Load parallel sentences from two text files (one sentence per line).
    Maps:
      - 'english'     -> German sentences (anchor, teacher will encode this)
      - 'non_english' -> other language sentences
    """
    logger.info(f"Loading parallel data from {source_file} and {target_file}")

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
    return Dataset.from_dict({"english": german_sentences, "non_english": other_sentences})


def split_train_eval(full_dataset: Dataset, name: str):
    eval_size = min(1000, int(len(full_dataset) * 0.1))
    split = full_dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
    logger.info(f"{name}: Train={len(split['train'])}, Eval={len(split['test'])}")
    return split["train"], split["test"]


# ============================================================================
# Initialize models
# ============================================================================

logger.info("Initializing teacher and student models")

teacher_model = SentenceTransformer(teacher_model_name)

def prepare_dataset(batch):
    """Encode German anchor sentences with teacher as distillation labels."""
    return {
        "english": batch["english"],
        "non_english": batch["non_english"],
        "label": teacher_model.encode(
            batch["english"],
            batch_size=inference_batch_size,
            show_progress_bar=False,
        ),
    }


def make_evaluator(eval_dataset: Dataset, lang_pair: str):
    return SequentialEvaluator(
        [
            MSEEvaluator(
                source_sentences=eval_dataset["english"],
                target_sentences=eval_dataset["non_english"],
                name=lang_pair,
                teacher_model=teacher_model,
                batch_size=inference_batch_size,
            ),
            TranslationEvaluator(
                source_sentences=eval_dataset["english"],
                target_sentences=eval_dataset["non_english"],
                name=lang_pair,
                batch_size=inference_batch_size,
            ),
        ],
        main_score_function=lambda scores: float(np.mean(scores)),
    )


def run_training(student_model, train_dataset, eval_dataset, num_epochs, output_dir, run_name_tag):
    """Train student model and save to output_dir/final."""
    labeled_train = train_dataset.map(
        prepare_dataset, batched=True, batch_size=30_000,
        remove_columns=train_dataset.column_names,
    )
    labeled_eval = eval_dataset.map(
        prepare_dataset, batched=True, batch_size=30_000,
        remove_columns=eval_dataset.column_names,
    )
    lang_pair = run_name_tag.split("_")[-1]  # e.g. "cs-de" from "stage1_cs-de"
    evaluator = make_evaluator(eval_dataset, lang_pair)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
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
        run_name=run_name_tag,
    )

    trainer = SentenceTransformerTrainer(
        model=student_model,
        args=training_args,
        train_dataset=labeled_train,
        eval_dataset=labeled_eval,
        loss=MSELoss(model=student_model),
        evaluator=evaluator,
    )

    logger.info(f"***** Start training: {run_name_tag} ({num_epochs} epochs) *****")
    trainer.train()

    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    student_model.save(str(final_dir))
    logger.info(f"Model saved to {final_dir}")
    return final_dir


# ============================================================================
# Stage 1: cs-de proxy distillation (two-stage mode only)
# ============================================================================

if mode == "two-stage":
    if args.stage1_model_path:
        # Reuse a pre-trained Stage 1 model — skip training
        stage1_final_dir = Path(args.stage1_model_path)
        logger.info(f"Skipping Stage 1 training — loading pre-trained model from {stage1_final_dir}")
    else:
        logger.info("=" * 80)
        logger.info(f"Stage 1: Training on cs-de | max_sentences={stage1_max_sentences} | epochs={stage1_epochs}")
        logger.info("=" * 80)

        cs_de_full = load_parallel_data(
            data_dir / "Europarl.cs-de.cs",
            data_dir / "Europarl.cs-de.de",
            max_sentences=stage1_max_sentences,
        )
        cs_de_train, cs_de_eval = split_train_eval(cs_de_full, "cs-de")

        student_model = SentenceTransformer(student_model_name)
        student_model.max_seq_length = student_max_seq_length

        stage1_final_dir = run_training(
            student_model,
            cs_de_train,
            cs_de_eval,
            num_epochs=stage1_epochs,
            output_dir=run_root / "stage1_cs-de",
            run_name_tag=f"{run_name}_stage1_cs-de",
        )

    # Load Stage 1 model as starting point for Stage 2
    student_model = SentenceTransformer(str(stage1_final_dir))
    student_model.max_seq_length = student_max_seq_length
    logger.info(f"Stage 2 starting from Stage 1 model: {stage1_final_dir}")

else:
    # Single-stage: start from the original pretrained student
    logger.info("=" * 80)
    logger.info(f"Single-stage mode: training directly on {target_lang}-de (no proxy stage)")
    logger.info("=" * 80)
    student_model = SentenceTransformer(student_model_name)
    student_model.max_seq_length = student_max_seq_length

# ============================================================================
# Stage 2: target language fine-tuning
# ============================================================================

logger.info("=" * 80)
logger.info(f"Stage 2: Training on {target_lang}-de | epochs={stage2_epochs}")
logger.info("=" * 80)

target_files = target_data_files[target_lang]
target_full = load_parallel_data(
    target_files["source_file"],
    target_files["target_file"],
    max_sentences=None,
)
target_train, target_eval = split_train_eval(target_full, f"{target_lang}-de")

final_output_dir = run_training(
    student_model,
    target_train,
    target_eval,
    num_epochs=stage2_epochs,
    output_dir=run_root / f"stage2_{target_lang}-de",
    run_name_tag=f"{run_name}_stage2_{target_lang}-de",
)

logger.info("=" * 80)
logger.info(f"Training completed! Final model: {final_output_dir}")
logger.info("=" * 80)
