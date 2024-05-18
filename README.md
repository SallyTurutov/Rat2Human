## Rat2Human Model

The Rat2Human model is designed to optimize molecular activity from rats to human tissues. By leveraging pre-trained and fine-tuned transformer architectures, the model can enhance the activity of molecules across various tissues, addressing a significant challenge in drug development.

## Step-by-Step Instructions

### Preparing the Datasets

1. **Pair the Dataset:**
   ```bash
   python pair_dataset.py data/unpaired_data data/paired_data.csv 0
   ```
2. **Preprocess the Paired Data:**
   ```bash
   python preprocess.py data/paired_data.csv
   ```
3. **Prepare the MMP Data:**
   ```bash
   python preprocess_mmp.py data/mmp_prop.csv data/pretrain.csv data/vocab.pkl
   ```

   Alternatively, you can unzip the provided data directory and use it as it is.

### Training and Pre-training the Model

You can skip this section by using the checkpoint available: one for the model after [pre-training](https://drive.google.com/file/d/1uwpTL2GhRh_nb5uheZojqgb1qLSVn4TI/view?usp=sharing) and one for the [final Rat2Human model](https://drive.google.com/file/d/1MkKxZWvtqbuj_CzHLfNUczZ8QWilEyvi/view?usp=sharing).

4. **Train the Model:**
   ```bash
   python train.py --data-path data --save-directory trained/Transformer/ --model-choice transformer
   ```

### Evaluating the Model

You can also find the evaluated model with all the baselines and ablations in the `final_results` directory.

5. **Generate Predictions:**
   ```bash
   python generate.py --model-choice transformer --data-path data --test-file-name test --model-path experiments/trained/Transformer/checkpoint --save-directory experiments/evaluation_transformer --vocab-path data/vocab.pkl --epoch 85 --batch-size 1
   ```
6. **Evaluate the Generated Molecules:**
   ```bash
   python evaluate.py --data-path experiments/evaluation_transformer/test/evaluation_85/generated_molecules.csv
   ```

By following these steps, you can prepare the datasets, train the model, and evaluate its performance. The provided checkpoints and evaluation results allow for a streamlined process, making it easier to replicate and build upon the Rat2Human model's capabilities.
