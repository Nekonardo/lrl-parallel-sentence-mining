#!/usr/bin/env python3
"""Run bilingual nearest-neighbor similarity mining for all language pairs and models
defined in a YAML config.

Usage (from project root):
    python scripts/run_similarity.py configs/pipeline_config.yaml
    python scripts/run_similarity.py configs/pipeline_config.yaml --dry_run
"""

import argparse
import os
import subprocess
import sys

import yaml


def get_alias(model):
    return model.get('alias', model['name'])


def main():
    parser = argparse.ArgumentParser(description='Run similarity mining pipeline from config.')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing them')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    output_base = config['output_base_dir']
    skip_existing = config.get('skip_existing', True)
    sim_cfg = config.get('similarity', {})

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mining_script = os.path.join(script_dir, '../third_party/PaSeMiLL/code/scripts/bilingual_nearest_neighbor.py')

    total = skipped = failed = 0

    for model in config['models']:
        model_alias = get_alias(model)

        for pair_entry in config['language_pairs']:
            pair = pair_entry['pair']
            splits = pair_entry['splits']
            src_lang, tgt_lang = pair.split('-')  # e.g. 'hsb-de' -> 'hsb', 'de'

            for split in splits:
                vec_dir = os.path.join(output_base, pair, model_alias)
                src_vec = os.path.join(vec_dir, f'{pair}.{split}.{src_lang}.{model_alias}.vec')
                tgt_vec = os.path.join(vec_dir, f'{pair}.{split}.{tgt_lang}.{model_alias}.vec')
                output_file = os.path.join(vec_dir, f'{pair}.{split}.{model_alias}.sim')

                if not os.path.isfile(src_vec):
                    print(f'[SKIP] Source vec not found: {src_vec}')
                    skipped += 1
                    continue
                if not os.path.isfile(tgt_vec):
                    print(f'[SKIP] Target vec not found: {tgt_vec}')
                    skipped += 1
                    continue

                if skip_existing and os.path.isfile(output_file):
                    print(f'[SKIP] Output exists: {output_file}')
                    skipped += 1
                    continue

                cmd = [
                    sys.executable, mining_script,
                    '-se', src_vec,
                    '-te', tgt_vec,
                    '-k', str(sim_cfg.get('k', 10)),
                    '-m', sim_cfg.get('method', 'csls'),
                    '--cslsknn', str(sim_cfg.get('csls_knn', 20)),
                    '--gpu', str(sim_cfg.get('gpu', 1)),
                    '-o', output_file,
                ]

                print(f'[RUN] {" ".join(cmd)}')
                total += 1

                if not args.dry_run:
                    result = subprocess.run(cmd)
                    if result.returncode != 0:
                        print(f'[ERROR] Failed (return code {result.returncode}): {output_file}')
                        failed += 1

    print(f'\nDone. ran={total}, skipped={skipped}, failed={failed}')


if __name__ == '__main__':
    main()