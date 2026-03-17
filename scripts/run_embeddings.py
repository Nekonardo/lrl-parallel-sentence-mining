#!/usr/bin/env python3
"""Run sentence embedding for all language pairs and models defined in a YAML config.

Usage (from project root):
    python scripts/run_embeddings.py configs/embedding_config.yaml
    python scripts/run_embeddings.py configs/embedding_config.yaml --dry_run
"""

import argparse
import os
import subprocess
import sys

import yaml


def get_alias(model):
    return model.get('alias', model['name'])


def resolve_sonar_lang(lang_map, lang):
    """Resolve FLORES-200 code from lang_map.
    Entries are plain strings (native) or dicts with code/proxy/note (proxy).
    Returns (flores_code, is_proxy, note).
    """
    entry = lang_map.get(lang, 'eng_Latn')
    if isinstance(entry, dict):
        return entry['code'], entry.get('proxy', False), entry.get('note', '')
    return entry, False, ''


def main():
    parser = argparse.ArgumentParser(description='Run embedding pipeline from config.')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing them')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    input_base = config['input_base_dir']
    output_base = config['output_base_dir']
    skip_existing = config.get('skip_existing', True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    embedding_script = os.path.join(script_dir, '../src/benchmark/contextual_sentence_embeddings.py')
    sonar_script = os.path.join(script_dir, '../src/benchmark/contextual_sentence_embeddings_sonar.py')

    total = skipped = failed = 0

    for model in config['models']:
        model_name = model['name']
        model_alias = get_alias(model)
        model_path = model.get('path')

        for pair_entry in config['language_pairs']:
            pair = pair_entry['pair']
            splits = pair_entry['splits']
            langs = pair.split('-')  # e.g. 'hsb-de' -> ['hsb', 'de']

            for split in splits:
                for lang in langs:
                    input_file = os.path.join(input_base, pair, f'{pair}.{split}.{lang}')
                    output_dir = os.path.join(output_base, pair, model_alias)
                    output_file = os.path.join(output_dir, f'{pair}.{split}.{lang}.{model_alias}.vec')

                    if not os.path.isfile(input_file):
                        print(f'[SKIP] Input not found: {input_file}')
                        skipped += 1
                        continue

                    if skip_existing and os.path.isfile(output_file):
                        print(f'[SKIP] Output exists: {output_file}')
                        skipped += 1
                        continue

                    os.makedirs(output_dir, exist_ok=True)

                    if model_name == 'sonar':
                        lang_map = model.get('lang_map', {})
                        source_lang, is_proxy, proxy_note = resolve_sonar_lang(lang_map, lang)
                        if is_proxy:
                            print(f'[WARN] SONAR proxy for "{lang}": {source_lang}. {proxy_note}')
                        cmd = [
                            sys.executable, sonar_script,
                            '-i', input_file,
                            '-o', output_file,
                            '-l', source_lang,
                        ]
                    else:
                        cmd = [
                            sys.executable, embedding_script,
                            '-i', input_file,
                            '-o', output_file,
                            '-m', model_name,
                        ]
                        if model_path is not None:
                            cmd += ['--pretrained_path', model_path]

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
