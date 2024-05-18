# coding=utf-8

"""
Implementation of a SMILES dataset.
"""
import pandas as pd

import torch
import torch.utils.data as tud
from torch.autograd import Variable
from typing import List

import configuration.config_default as cfgd
from models.transformer.module.subsequent_mask import subsequent_mask
from models.dataset_utils import load_data_from_smiles, get_molecules_features

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class Molecule:
    """
        Class that represents a train/validation/test datum
    """

    def __init__(self, smiles, encoded, tissue):
        self.smiles = smiles
        self.tissue = tissue

        self.encoded = torch.tensor(encoded, dtype=torch.long)

        node_features, adj_matrix, distances_matrix = load_data_from_smiles(smiles)
        self.node_features = node_features
        self.adj_matrix = adj_matrix
        self.distances_matrix = distances_matrix


class Dataset(tud.Dataset):
    "Custom PyTorch Dataset that takes a file containing Source_Mol, Target_Mol, Tissue, Source_Organism, Target_Organism."

    def __init__(self, data, vocabulary, tokenizer, prediction_mode=False):
        """
        :param data: dataframe read from training, validation or test file
        :param vocabulary: used to encode source/target tokens
        :param tokenizer: used to tokenize source/target smiles
        :param prediction_mode: if use target smiles or not (training or test)
        """
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._data = data
        self._prediction_mode = prediction_mode

    def __getitem__(self, i):
        "Tokenize and encode source smile and/or target smile (if prediction_mode is True)"

        row = self._data.iloc[i]

        tissue = row['Tissue']

        # tokenize and encode source smiles
        source_smi = row['Source_Mol']
        source_tokens = self._tokenizer.tokenize(source_smi)
        source_encoded = self._vocabulary.encode(source_tokens)
        source_molecule = Molecule(source_smi, source_encoded, tissue)

        # tokenize and encode target smiles if it is for training instead of evaluation
        if not self._prediction_mode:
            target_smi = row['Target_Mol']
            target_tokens = self._tokenizer.tokenize(target_smi)
            target_encoded = self._vocabulary.encode(target_tokens)
            target_molecule = Molecule(target_smi, target_encoded, tissue)

            return source_molecule, target_molecule, row
        else:
            return source_molecule, row

    def __len__(self):
        return len(self._data)


    @classmethod
    def _mol_to_collated_arr(cls, molecules: List[Molecule]):
        max_length = max([seq.encoded.size(0) for seq in molecules])
        collated_arr = torch.zeros(len(molecules), max_length, dtype=torch.long)
        for i, seq in enumerate(molecules):
            collated_arr[i, :seq.encoded.size(0)] = seq.encoded
        return collated_arr, max_length

    @classmethod
    def collate_fn(cls, data_all):
        data_all.sort(key=lambda x: len(x[0].encoded), reverse=True)
        is_prediction_mode = True if len(data_all[0]) == 2 else False
        if is_prediction_mode:
            source_molecules, data = zip(*data_all)
            data = pd.DataFrame(data)

            trg, trg_mask = None, None
        else:
            source_molecules, target_molecules, data = zip(*data_all)
            data = pd.DataFrame(data)

            trg, trg_max_length = cls._mol_to_collated_arr(target_molecules)
            trg_mask = (trg != 0).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).type_as(trg_mask))
            trg_mask = trg_mask[:, :-1, :-1]  # save start token, skip end token

        tissues_list = [cfgd.TISSUES[molecule.tissue] for molecule in source_molecules]

        src, src_max_length = cls._mol_to_collated_arr(source_molecules)
        src_mask = (src != 0).unsqueeze(-2)
        src_adj_matrix, src_node_features, src_distances_matrix = get_molecules_features(source_molecules, src_max_length)

        return src, trg, src_mask, trg_mask, src_adj_matrix, src_distances_matrix, tissues_list, data
