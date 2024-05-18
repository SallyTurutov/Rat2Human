import csv
import os
from tqdm import tqdm
import argparse
from typing import List, Dict, Any

from utils.chem import tanimoto_similarity


def process_csv_files(input_dir: str, output_file: str, similarity_threshold: float) -> None:
    """
    Process CSV files in the input directory to identify valid pairs of source and target molecules
    meeting specified conditions and write them to the output CSV file.

    Args:
    input_dir: Path to the directory containing input CSV files.
    output_file: Path to the output CSV file to be generated.
    similarity_threshold: Threshold value for similarity score between source and target molecules.
    """
    output_data = []
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".csv"):
            tissue_name = os.path.splitext(filename)[0]
            csv_data = read_csv(os.path.join(input_dir, filename))
            valid_pairs = find_valid_pairs(csv_data, similarity_threshold)
            for pair in valid_pairs:
                pair['Tissue'] = tissue_name
                output_data.append(pair)
    if output_data:
        write_to_csv(output_data, output_file)


def read_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Read data from a CSV file and return it as a list of dictionaries.

    Args:
    file_path: Path to the input CSV file.

    Returns:
    List of dictionaries representing data rows.
    """
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def find_valid_pairs(data: List[Dict[str, str]], similarity_threshold: float) -> List[Dict[str, str]]:
    """
    Find valid pairs of source and target molecules meeting specified conditions.

    Args:
    data: List of dictionaries representing data rows.
    similarity_threshold: Threshold value for similarity score between source and target molecules.

    Returns:
    List of dictionaries representing valid pairs.
    """
    valid_pairs = []
    for target_row in tqdm(data):
        if target_row['Human'] == '1':
            target_smiles = target_row['SMILES']
            max_similarity = 0
            best_value = None
            for source_row in data:
                if source_row['Rat'] == target_row['Human']:
                    label = target_row['Human']
                    source_smiles = source_row['SMILES']
                    similarity_score = calculate_similarity(source_smiles, target_smiles)
                    if similarity_score > similarity_threshold and similarity_score > max_similarity:
                        max_similarity = similarity_score
                        best_value = {
                            'Source_Mol': source_smiles,
                            'Target_Mol': target_smiles,
                            'Source_Organism': 'Rat',
                            'Target_Organism': 'Human',
                            'Label': label
                        }
            if best_value is not None:
                valid_pairs.append(best_value)
    return valid_pairs


def calculate_similarity(source_smiles: str, target_smiles: str) -> float:
    """
    Calculate similarity score between source and target molecules based on their SMILES representations.
    This function needs to be implemented separately.

    Args:
    source_smiles: SMILES representation of the source molecule.
    target_smiles: SMILES representation of the target molecule.

    Returns:
    Similarity score between source and target molecules.
    """
    similarity_score = tanimoto_similarity(source_smiles, target_smiles)
    return 0 if similarity_score is None else similarity_score


def write_to_csv(data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Write data to a CSV file.

    Args:
    data: List of dictionaries representing data rows.
    output_file: Path to the output CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Source_Mol', 'Target_Mol', 'Tissue', 'Source_Organism', 'Target_Organism', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(
        description="Process CSV files to identify valid pairs of source and target molecules.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing input CSV files.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file to be generated.")
    parser.add_argument("similarity_threshold", type=float,
                        help="Threshold value for similarity score between source and target molecules.")
    args = parser.parse_args()

    process_csv_files(args.input_dir, args.output_file, args.similarity_threshold)


if __name__ == "__main__":
    main()

# Run using: python pair_dataset.py data/unpaired_data data/paired_data.csv 0.5
