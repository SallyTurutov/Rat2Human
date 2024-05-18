import os
import csv
import argparse
import pickle
import pandas as pd
from tqdm import tqdm
from typing import Set, Dict

from BioMolX.predict import activity_prediction
from preprocess.vocabulary import Vocabulary, SMILESTokenizer, create_vocabulary
from preprocess.data_preparation import split_data, get_smiles_list
from utils.file import get_parent_dir

tissue_list = ['brain', 'kidney', 'prostate']


def preprocess_data(input_data_path: str, output_data_path: str) -> None:
    df = pd.read_csv(input_data_path)

    for tissue in tqdm(tissue_list):
        # Get all source and target molecules for the current tissue
        source_mols = df['Source_Mol'].tolist()
        target_mols = df['Target_Mol'].tolist()

        # Get predictions for source and target molecules
        source_predictions = activity_prediction(tissue, source_mols)
        target_predictions = activity_prediction(tissue, target_mols)

        # Assign predictions to DataFrame columns
        df[f'{tissue}_source_mol'] = source_predictions
        df[f'{tissue}_target_mol'] = target_predictions

    df.to_csv(output_data_path, index=False)


def choose_tissue(output_data_path: str) -> None:
    df = pd.read_csv(output_data_path)

    # Get list of tissue columns
    tissue_cols = [f'{tissue}_target_mol' for tissue in tissue_list]

    def get_column(row, value):
        for col_name, col_value in row.items():
            if col_value == value:
                return col_name

    def get_max_tissue(row):
        max_value_col = max(row[tissue_cols])
        tissue_name = get_column(row, max_value_col).split('_')[0]
        return tissue_name

    # Find the tissue with maximum value for each row
    df['Tissue'] = df.apply(get_max_tissue, axis=1)

    # Add Label column with constant value of 1
    df['Label'] = 1

    df.to_csv('data/pretrain.csv', index=False)



def load_vocabulary(vocab_file: str) -> Vocabulary:
    with open(vocab_file, 'rb') as f:
        vocabulary = pickle.load(f)
    return vocabulary

def add_to_vocabulary(vocabulary: Vocabulary, data_file: str) -> None:
    tokenizer = SMILESTokenizer()
    smiles_list = get_smiles_list(data_file)

    tokens = set()
    for smiles in smiles_list:
        tokens.update(tokenizer.tokenize(smiles, with_begin_and_end=False))
    vocabulary.update(sorted(tokens))


def save_vocabulary(vocabulary: Vocabulary, vocab_file: str) -> None:
    with open(vocab_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)


def main() -> None:
    """
    Main function to preprocess the data.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess CSV data to update existing vocabulary and split into train, validation, and test sets.")
    parser.add_argument("input_data_path", default="data/mmp_prop.csv", type=str, help="Path to the input data CSV.")
    parser.add_argument("output_data_path", default="data/pretrain.csv", type=str, help="Path to the output data CSV.")
    parser.add_argument("vocab_file", default="data/vocab.pkl", type=str, help="Path to the existing vocabulary pickle file.")
    args = parser.parse_args()

    preprocess_data(args.input_data_path, args.output_data_path)
    choose_tissue(args.output_data_path)
    vocabulary = load_vocabulary(args.vocab_file)
    add_to_vocabulary(vocabulary, args.output_data_path)
    save_vocabulary(vocabulary, args.vocab_file)
    print(f"vocabulary: {vocabulary.tokens()}")


if __name__ == "__main__":
    main()

