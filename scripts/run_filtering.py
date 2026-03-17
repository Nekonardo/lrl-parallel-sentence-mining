#!/usr/bin/env python3
"""Threshold optimization for bilingual sentence pair filtering.

For each (lang_pair, model):
  - Sweep thresholds on both train and test splits
  - Find best threshold on train
  - Apply train-best threshold to test and compute transfer gap

Outputs:
  {output_base}/filtering_sweep.csv     — full P/R/F1 for every threshold × split × config
  {output_base}/filtering_summary.csv   — one row per (lang_pair, model) with best thresholds and gap
  {vec_dir}/{pair}.{split}.{model}.pred.best       — predictions at split-optimal threshold
  {vec_dir}/{pair}.test.{model}.pred.train_th      — test predictions at train-optimal threshold

Usage (from project root):
    python scripts/run_filtering.py configs/pipeline_config.yaml
    python scripts/run_filtering.py configs/pipeline_config.yaml --dry_run
"""

import argparse
import csv
import os
import tempfile

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Inlined filter logic (from third_party/PaSeMiLL/code/scripts/filter.py)
# ---------------------------------------------------------------------------

def _apply_filter(sim_file, pred_file, method, threshold):
    """Filter sim_file by threshold, write pairs above threshold to pred_file."""
    if method == 'dynamic':
        scores = []
        with open(sim_file, 'r') as f:
            for line in f:
                scores.append(float(line.split('\t')[2]))
        s = np.array(scores)
        threshold = s.mean() + threshold * s.std()

    with open(sim_file, 'r') as fin, open(pred_file, 'w') as fout:
        for line in fin:
            parts = line.split('\t')
            if float(parts[2]) > threshold:
                fout.write(f'{parts[0]}\t{parts[1]}\n')


# ---------------------------------------------------------------------------
# Inlined evaluation logic (from third_party/PaSeMiLL/code/scripts/bucc_f-score.py)
# ---------------------------------------------------------------------------

def _evaluate(pred_file, gold_file):
    """Return (precision, recall, f1) by comparing pred to gold."""
    gold = {}
    with open(gold_file, 'r') as f:
        for line in f:
            src, tgt = line.rstrip('\n').split('\t')
            gold[src] = tgt

    tp = fp = 0
    seen = set()
    with open(pred_file, 'r') as f:
        for line in f:
            src, tgt = line.rstrip('\n').split('\t')
            seen.add(src)
            if src in gold and tgt == gold[src]:
                tp += 1
            else:
                fp += 1

    fn = len(gold) - tp
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1


# ---------------------------------------------------------------------------
# Threshold sweep for one sim file
# ---------------------------------------------------------------------------

def _sweep(sim_file, gold_file, method, start, end, step):
    """Return list of dicts {threshold, precision, recall, f1}."""
    results = []
    threshold = start
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pred', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        while threshold <= end + 1e-9:
            _apply_filter(sim_file, tmp_path, method, threshold)
            P, R, F1 = _evaluate(tmp_path, gold_file)
            results.append({'threshold': round(threshold, 10), 'precision': P, 'recall': R, 'f1': F1})
            threshold += step
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return results


def get_alias(model):
    return model.get('alias', model['name'])


def main():
    parser = argparse.ArgumentParser(description='Run threshold optimization from config.')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be done without running')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_base  = config['input_base_dir']
    output_base = config['output_base_dir']
    skip_existing = config.get('skip_existing', True)
    flt = config.get('filtering', {})
    method = flt.get('method', 'dynamic')
    th_start = flt.get('threshold_start', 0.1)
    th_end   = flt.get('threshold_end',   5.0)
    th_step  = flt.get('threshold_step',  0.1)

    sweep_rows   = []   # all threshold × split × config rows
    summary_rows = []   # one row per (lang_pair, model)

    for model in config['models']:
        model_alias = get_alias(model)

        for pair_entry in config['language_pairs']:
            pair   = pair_entry['pair']
            splits = pair_entry['splits']

            vec_dir   = os.path.join(output_base, pair, model_alias)
            gold_base = os.path.join(input_base, pair)

            # Collect sweep results keyed by split
            sweep_by_split = {}

            for split in splits:
                sim_file  = os.path.join(vec_dir, f'{pair}.{split}.{model_alias}.sim')
                gold_file = os.path.join(gold_base, f'{pair}.{split}.gold')

                if not os.path.isfile(sim_file):
                    print(f'[SKIP] sim not found:  {sim_file}')
                    continue
                if not os.path.isfile(gold_file):
                    print(f'[SKIP] gold not found: {gold_file}')
                    continue

                print(f'[SWEEP] {pair} | {split} | {model_alias}  ({th_start}→{th_end} step {th_step})')
                if args.dry_run:
                    continue

                rows = _sweep(sim_file, gold_file, method, th_start, th_end, th_step)
                sweep_by_split[split] = rows

                for r in rows:
                    sweep_rows.append({
                        'lang_pair': pair,
                        'model':     model_alias,
                        'split':     split,
                        **r,
                    })

                # Save per-split best pred file
                best = max(rows, key=lambda x: x['f1'])
                best_pred = os.path.join(vec_dir, f'{pair}.{split}.{model_alias}.pred.best')
                if not (skip_existing and os.path.isfile(best_pred)):
                    _apply_filter(sim_file, best_pred, method, best['threshold'])
                    print(f'  → best pred ({split}): threshold={best["threshold"]:.2f}  '
                          f'F1={best["f1"]:.4f}  saved to {best_pred}')

            if args.dry_run or 'train' not in sweep_by_split:
                continue

            # ------------------------------------------------------------------
            # Transfer analysis: apply train-best threshold to test
            # ------------------------------------------------------------------
            train_rows = sweep_by_split['train']
            train_best = max(train_rows, key=lambda x: x['f1'])
            train_best_th = train_best['threshold']

            summary = {
                'lang_pair':         pair,
                'model':             model_alias,
                'train_best_th':     train_best_th,
                'train_best_f1':     train_best['f1'],
                'train_best_p':      train_best['precision'],
                'train_best_r':      train_best['recall'],
            }

            if 'test' in sweep_by_split:
                test_rows = sweep_by_split['test']
                test_best = max(test_rows, key=lambda x: x['f1'])

                # Find test result at train-best threshold (nearest step)
                test_at_train = min(test_rows, key=lambda x: abs(x['threshold'] - train_best_th))

                summary.update({
                    'test_best_th':      test_best['threshold'],
                    'test_best_f1':      test_best['f1'],
                    'test_best_p':       test_best['precision'],
                    'test_best_r':       test_best['recall'],
                    'test_f1_at_train_th':  test_at_train['f1'],
                    'test_p_at_train_th':   test_at_train['precision'],
                    'test_r_at_train_th':   test_at_train['recall'],
                    'transfer_gap':      test_best['f1'] - test_at_train['f1'],
                })

                # Save test pred at train-best threshold
                sim_test = os.path.join(vec_dir, f'{pair}.test.{model_alias}.sim')
                pred_train_th = os.path.join(vec_dir, f'{pair}.test.{model_alias}.pred.train_th')
                if not (skip_existing and os.path.isfile(pred_train_th)):
                    _apply_filter(sim_test, pred_train_th, method, train_best_th)
                    print(f'  → test@train_th: threshold={train_best_th:.2f}  '
                          f'F1={test_at_train["f1"]:.4f}  gap={summary["transfer_gap"]:.4f}  '
                          f'saved to {pred_train_th}')

            summary_rows.append(summary)

    if args.dry_run:
        print('\n[dry_run] No files written.')
        return

    # --------------------------------------------------------------------------
    # Write CSVs
    # --------------------------------------------------------------------------
    os.makedirs(output_base, exist_ok=True)

    sweep_csv = os.path.join(output_base, 'filtering_sweep.csv')
    if sweep_rows:
        with open(sweep_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['lang_pair', 'model', 'split',
                                                    'threshold', 'precision', 'recall', 'f1'])
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f'\n[CSV] Sweep results → {sweep_csv}')

    summary_csv = os.path.join(output_base, 'filtering_summary.csv')
    if summary_rows:
        fields = [
            'lang_pair', 'model',
            'train_best_th', 'train_best_f1', 'train_best_p', 'train_best_r',
            'test_best_th',  'test_best_f1',  'test_best_p',  'test_best_r',
            'test_f1_at_train_th', 'test_p_at_train_th', 'test_r_at_train_th',
            'transfer_gap',
        ]
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f'[CSV] Summary         → {summary_csv}')

        # Print summary table to console
        print(f'\n{"Lang Pair":<12} {"Model":<24} {"TrainTh":>8} {"TrainF1":>8} '
              f'{"TestF1@TrainTh":>15} {"TestBestF1":>11} {"Gap":>8}')
        print('-' * 95)
        for r in sorted(summary_rows, key=lambda x: x.get('test_best_f1', 0), reverse=True):
            print(f'{r["lang_pair"]:<12} {r["model"]:<24} '
                  f'{r["train_best_th"]:>8.2f} {r["train_best_f1"]:>8.4f} '
                  f'{r.get("test_f1_at_train_th", float("nan")):>15.4f} '
                  f'{r.get("test_best_f1", float("nan")):>11.4f} '
                  f'{r.get("transfer_gap", float("nan")):>8.4f}')


if __name__ == '__main__':
    main()