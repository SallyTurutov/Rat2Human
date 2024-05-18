import argparse
import pandas as pd
import numpy as np
import os
import statistics
from multiprocessing import Pool
import math

import utils.log as ul
import utils.chem as uc
import configuration.config_default as cfgd
import utils.file as uf
import utils.plot as up
import configuration.opts as opts
from BioMolX.predict import activity_prediction

NUM_WORKERS = 16


class EvaluationRunner:
    "Evaluate the generated molecules"

    def __init__(self, data_path, num_samples, range_evaluation):

        self.save_path = uf.get_parent_dir(data_path)
        global LOG
        LOG = ul.get_logger(name="evaluation", log_path=os.path.join(self.save_path, 'evaluation.log'))
        self.data_path = data_path
        self.data = pd.read_csv(self.data_path, sep=",")
        self.num_samples = num_samples

        self.output_path = self.save_path
        self.range_evaluation = range_evaluation
        if self.range_evaluation != "":
            self.output_path = os.path.join(self.output_path, '{}'.format(self.range_evaluation))
        uf.make_directory(self.output_path)

    def evaluation_statistics(self):
        # Compute Metrics
        self.compute_similarity()
        self.compute_activity()

        # Save to file
        statistics_file = self.data_path.split(".csv")[0] + "_statistics.csv"
        self.data.to_csv(statistics_file, index=False)

        best_file = os.path.join(self.output_path, 'best_optimized.csv')
        self.create_best_optimized(statistics_file, best_file)

        self.compute_novelty(best_file)
        self.compute_diversity(best_file)

    def compute_similarity(self):
        LOG.info('Computing Tanimoto similarity')
        source_smiles_list = self.data['Source_Mol'].tolist()
        similarities = []
        for i in range(self.num_samples):
            pred_smi_list = self.data['Predicted_smi_{}'.format(i + 1)].tolist()
            zipped = list(zip(source_smiles_list, pred_smi_list))
            with Pool(NUM_WORKERS) as p:
                results = p.map(uc.tanimoto_similarity_pool, zipped)
            similarities.extend(results)
            self.data[f'Predicted_smi_{i + 1}_tanimoto'] = results

        results_not_none = [s for s in similarities if s]
        up.hist_box_list(results_not_none, name="similarity",
                         path=self.output_path, title="Similarity")
        LOG.info(f'Tanimoto Mean: {statistics.mean(results_not_none)}')

    def compute_activity(self):
        LOG.info('Computing Tissue Activity')
        tissue_list = self.data['Tissue'].tolist()
        activities = []
        for i in range(self.num_samples):
            pred_smi_list = self.data['Predicted_smi_{}'.format(i + 1)].tolist()
            zipped = list(zip(tissue_list, pred_smi_list))
            results = []
            for tissue, smiles in zipped:
                results.append(activity_prediction(tissue, smiles))
            activities.extend(results)
            self.data[f'Predicted_tissue_activity_{i + 1}'] = results

        results_not_none = [s for s in activities if s]
        up.hist_box_list(results_not_none, name="Tissue Activity",
                         path=self.output_path, title="Tissue Activity")
        LOG.info(f'Tissue Activity Mean: {statistics.mean(results_not_none)}')

    def compute_novelty(self, out_file):
        LOG.info('Computing Novelty')

        pretrain_data = pd.read_csv('data/pretrain.csv')
        val_data = pd.read_csv('data/train.csv')
        train_data = pd.read_csv('data/validation.csv')
        test_data = pd.read_csv('data/test.csv')

        all_data = pd.concat([pretrain_data, train_data, val_data, test_data], ignore_index=True)
        all_molecules = set(all_data['Source_Mol']).union(set(all_data['Target_Mol']))

        data = pd.read_csv(out_file, sep=",")
        optimized_smiles_list = data['Optimized_Mol'].tolist()

        novelties = []
        for mol in optimized_smiles_list:
            novelty_score = 0 if mol in all_molecules else 1
            novelties.append(novelty_score)
        data['Novelty'] = novelties
        data.to_csv(out_file, index=False)

        novelty_mean = sum(novelties) / len(novelties) if novelties else 0
        LOG.info(f'Novelty Mean: {novelty_mean}')

    def compute_diversity(self, out_file):
        LOG.info('Computing Diversity')
        data = pd.read_csv(out_file, sep=",")
        optimized_smiles_list = data['Optimized_Mol'].tolist()
        tissues_list = data['Tissue'].tolist()

        mols_dict = {tissue: [] for tissue in set(tissues_list)}
        for tissue, mol in zip(tissues_list, optimized_smiles_list):
            mols_dict[tissue].append(mol)

        diversities_dict = {tissue: len(set(mols_dict[tissue])) / len(mols_dict[tissue]) for tissue in
                            set(tissues_list)}
        diversities = [diversities_dict[tissue] for tissue in tissues_list]

        data['Diversity'] = diversities
        data.to_csv(out_file, index=False)

        LOG.info(f'Diversity: {diversities_dict}')

    def create_best_optimized(self, in_file, out_file):
        best_optimized_data = []

        data = pd.read_csv(in_file, sep=",")
        for index, row in data.iterrows():
            source_mol = row['Source_Mol']
            tissue = row['Tissue']
            source_organism = row['Source_Organism']
            target_organism = row['Target_Organism']

            best_tanimoto_similarity = 0
            best_tissue_activity = 0
            best_optimized_mol = ""

            for i in range(self.num_samples):
                tanimoto_similarity = row[f'Predicted_smi_{i + 1}_tanimoto']
                tissue_activity = row[f'Predicted_tissue_activity_{i + 1}']
                predicted_mol = row[f'Predicted_smi_{i + 1}']

                if tanimoto_similarity > 0 and tissue_activity > best_tissue_activity:
                    best_tanimoto_similarity = tanimoto_similarity
                    best_tissue_activity = tissue_activity
                    best_optimized_mol = predicted_mol

            if best_optimized_mol:
                best_optimized_data.append({
                    'Source_Mol': source_mol,
                    'Optimized_Mol': best_optimized_mol,
                    'Tissue': tissue,
                    'Source_Organism': source_organism,
                    'Optimized_Organism': target_organism,
                    'Tanimoto_Similarity': best_tanimoto_similarity,
                    'Tissue_Activity': best_tissue_activity
                })

        best_optimized_df = pd.DataFrame(best_optimized_data)
        best_optimized_df.to_csv(out_file, index=False)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='evaluation.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opts.evaluation_opts(parser)
    opt = parser.parse_args()

    runner = EvaluationRunner(opt.data_path, opt.num_samples, opt.range_evaluation)
    runner.evaluation_statistics()


if __name__ == "__main__":
    main()
