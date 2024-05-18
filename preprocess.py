import os
import csv
import argparse
import pickle
from typing import Set, Dict

from preprocess.vocabulary import SMILESTokenizer, create_vocabulary
from preprocess.data_preparation import split_data, get_smiles_list
from utils.file import get_parent_dir


def make_vocabulary(data_file: str) -> Set[str]:
    """
    Build vocabulary from the data file.

    Args:
    data_file: Path to the data CSV file.

    Returns:
    Set of tokens in the vocabulary.
    """
    tokenizer = SMILESTokenizer()
    smiles_list = get_smiles_list(data_file)
    vocabulary = create_vocabulary(smiles_list, tokenizer=tokenizer)
    return vocabulary


def preprocess_data(data_path: str) -> None:
    """
    Preprocess data by building vocabulary and splitting it into train, validation, and test sets.

    Args:
    data_path: Path to the data CSV file.
    """
    parent_path = get_parent_dir(data_path)
    vocabulary = make_vocabulary(data_path)
    print(f"vocabulary: {vocabulary.tokens()}")
    output_file = os.path.join(parent_path, 'vocab.pkl')
    with open(output_file, 'wb') as pickled_file:
        pickle.dump(vocabulary, pickled_file)

    split_data(data_path)


def main() -> None:
    """
    Main function to parse command-line arguments and preprocess the data.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess CSV data to build vocabulary and split into train, validation, and test sets.")
    parser.add_argument("data_path", type=str, help="Path to the data CSV.")
    args = parser.parse_args()

    preprocess_data(args.data_path)


if __name__ == "__main__":
    main()
