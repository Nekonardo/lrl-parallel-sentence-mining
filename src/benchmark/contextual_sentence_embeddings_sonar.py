import argparse
import os
import sys
import torch
from tqdm import tqdm
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '../../third_party/PaSeMiLL/code'))
import utils as utils


def parse_args():
    parser = argparse.ArgumentParser(description='Generate sentence embeddings using SONAR')
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Sentence file (format: <ID>\\t<sentence>)')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Output file path')
    parser.add_argument('-l', '--source_lang', type=str, default='eng_Latn',
                        help='Source language code (default: eng_Latn). Examples: deu_Latn (German), fra_Latn (French), spa_Latn (Spanish)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto (default: auto)')

    return parser.parse_args()


def to_sonar_embeddings(path, sentence_list, source_lang="eng_Latn", device=None, start_i=0):
    """Save embeddings from SONAR text encoder in txt file (same format as fastText)."""
    # Set device
    if device is None or device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f'Loading SONAR text encoder (text_sonar_basic_encoder)')
    print(f'Using device: {device}')
    print(f'Source language: {source_lang}')

    # Initialize SONAR encoder
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )
    embedding_size = 1024  # SONAR embeddings are 1024-dimensional

    # Initial step
    if start_i == 0:
        sentence = sentence_list[0]
        split_sentence = sentence.split('\t')
        print(f'First sentence: {split_sentence}')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'

        # Get SONAR embedding
        embeddings = t2vec_model.predict([split_sentence[1]], source_lang=source_lang)
        np_embedding = embeddings.cpu().detach().numpy()[0]
        str_embedding = [f'{embed_value:.6f}' for embed_value in np_embedding]

        # First line
        n = len(sentence_list)
        vec_size = embedding_size

        with open(path, 'w', encoding='utf8') as out_text:
            out_text.write(f'{n} {vec_size}\n{split_sentence[0]} {" ".join(str_embedding)}\n')

    embedding_list = []
    for i in tqdm(range(start_i + 1, len(sentence_list)), desc="Encoding sentences"):
        sentence = sentence_list[i]
        split_sentence = sentence.split('\t')
        assert len(split_sentence) == 2, f'The line contains too many fields: {len(split_sentence)}'

        # Get SONAR embedding
        embeddings = t2vec_model.predict([split_sentence[1]], source_lang=source_lang)
        np_embedding = embeddings.cpu().detach().numpy()[0]
        str_embedding = [f'{embed_value:.6f}' for embed_value in np_embedding]

        if str_embedding:  # Not None
            embedding_list.append(f'{split_sentence[0]} {" ".join(str_embedding)}')
        if i % 10000 == 0:
            with open(path, 'a', encoding='utf8') as out_text:
                out_text.write('\n'.join(embedding_list) + '\n')
            embedding_list = []

    # Remaining lines
    with open(path, 'a', encoding='utf8') as out_text:
        out_text.write('\n'.join(embedding_list) + '\n')


def main():
    args = parse_args()

    # Input file
    print(f'Reading input file: {args.input_file}')
    input_file = open(args.input_file, 'r').read()
    split_file = utils.text_to_line(input_file)
    print(f'Total sentences: {len(split_file)}')

    # Generate embeddings
    to_sonar_embeddings(
        path=args.output_file,
        sentence_list=split_file,
        source_lang=args.source_lang,
        device=args.device,
        start_i=0
    )

    print(f'Embeddings saved to: {args.output_file}')


if __name__ == '__main__':
    main()