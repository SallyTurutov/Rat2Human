import os
from typing import List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.file import get_parent_dir

SEED = 42


def get_smiles_list(file_name: str) -> List[str]:
    "Get smiles list for building vocabulary."
    pd_data = pd.read_csv(file_name, sep=",")
    smiles_list = pd.unique(pd_data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
    return list(smiles_list)


def split_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    "Split data into training, validation, and test set, write to files."
    data = pd.read_csv(data_path, sep=",")

    train, test = train_test_split(data, test_size=0.1, random_state=SEED)
    train, validation = train_test_split(train, test_size=0.1, random_state=SEED)

    parent = get_parent_dir(data_path)
    train.to_csv(os.path.join(parent, "train.csv"), index=False)
    validation.to_csv(os.path.join(parent, "validation.csv"), index=False)
    test.to_csv(os.path.join(parent, "test.csv"), index=False)

    return train, validation, test
