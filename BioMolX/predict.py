import torch
from typing import List
from rdkit import Chem

from BioMolX.meta_model import Meta_model


def set_model(tissue: str) -> str:
    if tissue == "brain":
        return "BioMolX/checkpoints/01 - brain_model.pth"
    if tissue == "breast":
        return "BioMolX/checkpoints/02 - breast_model.pth"
    if tissue == "cervix":
        return "BioMolX/checkpoints/03 - cervix_model.pth"
    if tissue == "intestinal":
        return "BioMolX/checkpoints/04 - intestinal_model.pth"
    if tissue == "kidney":
        return "BioMolX/checkpoints/05 - kidney_model.pth"
    if tissue == "liver":
        return "BioMolX/checkpoints/06 - liver_model.pth"
    if tissue == "lung":
        return "BioMolX/checkpoints/07 - lung_model.pth"
    if tissue == "ovary":
        return "BioMolX/checkpoints/08 - ovary_model.pth"
    if tissue == "prostate":
        return "BioMolX/checkpoints/09 - prostate_model.pth"
    if tissue == "skin":
        return "BioMolX/checkpoints/10 - skin_model.pth"


def is_valid_molecule(smiles):
    """
    Check if a molecule represented by SMILES string is valid.

    Args:
        smiles: SMILES string representing the molecule.

    Returns:
        True if the molecule is valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return False

    return False if mol is None else True


def activity_prediction(tissue: str, smiles: str) -> int:
    if not is_valid_molecule(smiles):
        return None

    checkpoint_path = set_model(tissue)

    # Set up the device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Set up the model
    model_args = {
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "batch_size": 32,
        "lr": 0.001,
        "lr_scale": 1,
        "decay": 0,
        "num_layer": 5,
        "emb_dim": 300,
        "dropout_ratio": 0.5,
        "graph_pooling": "mean",
        "JK": "last",
        "gnn_type": "gin",
        "split": "scaffold",
        "eval_train": 0,
        "num_workers": 4,
        "num_tasks": 12,
        "n_way": 2,
        "m_support": 5,
        "k_query": 32,
        "meta_lr": 0.001,
        "update_lr": 0.4,
        "add_similarity": True,
        "add_selfsupervise": True,
        'add_masking': True,
        "add_weight": 0.1,
        "input_model_file": "BioMolX/model_gin/supervised_contextpred.pth",
    }

    model = Meta_model(model_args).to(model_args['device'])
    model = model.load_checkpoint(checkpoint_path)

    # Get predictions
    predictions = model.get_prediction([smiles])

    return predictions[0]


def activity_predictions(tissue: str, smiles_list: List[str]) -> List[int]:
    # if not is_valid_molecule(smiles):
    #     return None

    checkpoint_path = set_model(tissue)

    # Set up the device
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Set up the model
    model_args = {
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "batch_size": 32,
        "lr": 0.001,
        "lr_scale": 1,
        "decay": 0,
        "num_layer": 5,
        "emb_dim": 300,
        "dropout_ratio": 0.5,
        "graph_pooling": "mean",
        "JK": "last",
        "gnn_type": "gin",
        "split": "scaffold",
        "eval_train": 0,
        "num_workers": 4,
        "num_tasks": 12,
        "n_way": 2,
        "m_support": 5,
        "k_query": 32,
        "meta_lr": 0.001,
        "update_lr": 0.4,
        "add_similarity": True,
        "add_selfsupervise": True,
        'add_masking': True,
        "add_weight": 0.1,
        "input_model_file": "BioMolX/model_gin/supervised_contextpred.pth",
    }

    model = Meta_model(model_args).to(model_args['device'])
    model = model.load_checkpoint(checkpoint_path)

    # Get predictions
    return model.get_prediction(smiles_list)
