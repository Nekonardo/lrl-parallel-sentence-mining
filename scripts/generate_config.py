#!/usr/bin/env python3
"""Scan bucc_style_data directory and generate a YAML config for the full pipeline
(embedding + similarity mining).

# All pairs
python scripts/generate_config.py -o configs/pipeline_config.yaml

# Specific pairs only
python scripts/generate_config.py --pairs hsb-de dsb-de -o configs/hsb_dsb.yaml
"""

import argparse
import os
import yaml


def detect_language_pairs(input_base_dir):
    pairs = []
    if not os.path.isdir(input_base_dir):
        print(f'[WARN] input_base_dir not found: {input_base_dir}')
        return pairs
    for entry in sorted(os.listdir(input_base_dir)):
        pair_dir = os.path.join(input_base_dir, entry)
        if not os.path.isdir(pair_dir):
            continue
        splits = set()
        for fname in os.listdir(pair_dir):
            parts = fname.split('.')
            # Expected format: {pair}.{split}.{lang} or {pair}.{split}.gold
            if len(parts) == 3 and parts[2] != 'gold':
                splits.add(parts[1])
        if splits:
            pairs.append({'pair': entry, 'splits': sorted(splits)})
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Generate full pipeline config.')
    parser.add_argument('--input_base_dir', default='data/benchmark/bucc_style_data',
                        help='Root directory of bucc_style_data')
    parser.add_argument('--output_base_dir', default='output/embeddings',
                        help='Root directory for output .vec and .sim files')
    parser.add_argument('-o', '--output', default='configs/pipeline_config.yaml',
                        help='Path to write the generated config')
    parser.add_argument('--pairs', nargs='+', metavar='PAIR',
                        help='Restrict to specific language pairs, e.g. --pairs hsb-de dsb-de')
    args = parser.parse_args()

    language_pairs = detect_language_pairs(args.input_base_dir)
    print(f'Detected {len(language_pairs)} language pair(s): {[p["pair"] for p in language_pairs]}')

    if args.pairs:
        requested = set(args.pairs)
        language_pairs = [p for p in language_pairs if p['pair'] in requested]
        missing = requested - {p['pair'] for p in language_pairs}
        if missing:
            print(f'[WARN] Pairs not found in input_base_dir: {sorted(missing)}')
        print(f'Filtered to {len(language_pairs)} pair(s): {[p["pair"] for p in language_pairs]}')

    config = {
        'input_base_dir': args.input_base_dir,
        'output_base_dir': args.output_base_dir,
        'skip_existing': True,
        'models': [

            {'name': 'glot500'},
            {'name': 'xlmr'},
            {'name': 'labse'},
            {'name': 'laser2'},
            {'name': 'mmbert'},

            # sonar: dsb→Czech, chv→Tatar  (baseline proxy choices)
            {
                'name': 'sonar',
                'alias': 'sonar',
                'lang_map': {
                    'de':  'deu_Latn',
                    'en':  'eng_Latn',
                    'fr':  'fra_Latn',
                    'cs':  'ces_Latn',
                    'hsb': {'code': 'ces_Latn', 'proxy': True,
                            'note': 'hsb not supported; using Czech (ces_Latn) as proxy'},
                    'dsb': {'code': 'ces_Latn', 'proxy': True,
                            'note': 'dsb not supported; using Czech (ces_Latn) as proxy'},
                    'chv': {'code': 'tat_Cyrl', 'proxy': True,
                            'note': 'chv not supported; using Tatar (tat_Cyrl) as proxy'},
                },
            },
            # sonar-pol: dsb→Polish, chv→Tatar
            {
                'name': 'sonar',
                'alias': 'sonar-pol',
                'lang_map': {
                    'de':  'deu_Latn',
                    'en':  'eng_Latn',
                    'fr':  'fra_Latn',
                    'cs':  'ces_Latn',
                    'hsb': {'code': 'ces_Latn', 'proxy': True,
                            'note': 'hsb not supported; using Czech (ces_Latn) as proxy'},
                    'dsb': {'code': 'pol_Latn', 'proxy': True,
                            'note': 'dsb not supported; using Polish (pol_Latn) as proxy'},
                    'chv': {'code': 'tat_Cyrl', 'proxy': True,
                            'note': 'chv not supported; using Tatar (tat_Cyrl) as proxy'},
                },
            },
            # sonar-kaz: dsb→Czech, chv→Kazakh
            {
                'name': 'sonar',
                'alias': 'sonar-kaz',
                'lang_map': {
                    'de':  'deu_Latn',
                    'en':  'eng_Latn',
                    'fr':  'fra_Latn',
                    'cs':  'ces_Latn',
                    'hsb': {'code': 'ces_Latn', 'proxy': True,
                            'note': 'hsb not supported; using Czech (ces_Latn) as proxy'},
                    'dsb': {'code': 'ces_Latn', 'proxy': True,
                            'note': 'dsb not supported; using Czech (ces_Latn) as proxy'},
                    'chv': {'code': 'kaz_Cyrl', 'proxy': True,
                            'note': 'chv not supported; using Kazakh (kaz_Cyrl) as proxy'},
                },
            },
            # sonar-pol-kaz: dsb→Polish, chv→Kazakh
            {
                'name': 'sonar',
                'alias': 'sonar-pol-kaz',
                'lang_map': {
                    'de':  'deu_Latn',
                    'en':  'eng_Latn',
                    'fr':  'fra_Latn',
                    'cs':  'ces_Latn',
                    'hsb': {'code': 'ces_Latn', 'proxy': True,
                            'note': 'hsb not supported; using Czech (ces_Latn) as proxy'},
                    'dsb': {'code': 'pol_Latn', 'proxy': True,
                            'note': 'dsb not supported; using Polish (pol_Latn) as proxy'},
                    'chv': {'code': 'kaz_Cyrl', 'proxy': True,
                            'note': 'chv not supported; using Kazakh (kaz_Cyrl) as proxy'},
                },
            },
            # {
            #     'name': 'pretrained',
            #     'alias': 'pretrained-hsb-para-cs',
            #     'path': '../../pretraining_test/modelling/output-hsb-para-cs',
            # },
        ],
        'language_pairs': language_pairs,
        'similarity': {
            'k': 10,
            'method': 'csls',
            'csls_knn': 20,
            'gpu': 1,
        },
        'filtering': {
            'method': 'dynamic',   # dynamic or static
            'threshold_start': 0.1,
            'threshold_end': 5.0,
            'threshold_step': 0.1,
        },
    }

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f'Config written to {args.output}')


if __name__ == '__main__':
    main()