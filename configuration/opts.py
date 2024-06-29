""" Implementation of all available options """
from __future__ import print_function


def train_opts(parser):
    # Transformer
    parser.add_argument('--model-choice', required=True, help='transformer')

    # Common training options
    group = parser.add_argument_group('Training_options')
    group.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    group.add_argument('--num-epoch', type=int, default=85, help='Number of training steps')
    group.add_argument('--starting-epoch', type=int, default=1, help='Training from given starting epoch')
    group.add_argument('--use-data-parallel', help='Use pytorch DataParallel', action='store_true')
    group.add_argument('--pretrain-model', default=True, help='Whether to pretrain the model', action='store_true')
    group.add_argument('--use-pretrained', default=False, help='Whether to use pretrained model', action='store_true')
    group.add_argument('--num-pretrain-epoch', type=int, default=1, help='Number of training steps')

    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True, help='Input data path')
    group.add_argument('--save-directory', default='train', help='Result save directory')

    # Model architecture options
    group = parser.add_argument_group('Model')
    group.add_argument('-N', type=int, default=6, help='number of encoder and decoder')
    group.add_argument('-N_tissues', type=int, default=3, help='number of tissues in the dataset')
    group.add_argument('-H', type=int, default=8, help='heads of attention')
    group.add_argument('-d-model', type=int, default=256, help='embedding dimension, model dimension')
    group.add_argument('-d-ff', type=int, default=2048, help='dimension in feed forward network')
    group.add_argument('-d_atom', type=int, default=26, help='dimension of the molecule node_features')

    # Regularization
    group.add_argument('--dropout', type=float, default=0.1, help='dropout probability.')
    group.add_argument('--label-smoothing', type=float, default=0.0,
                       help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # Optimization options
    group = parser.add_argument_group('Optimization')
    group.add_argument('--factor', type=float, default=1.0,
                       help="""Factor multiplied to the learning rate scheduler formula in NoamOpt. 
                       For more information about the formula, 
                       see paper Attention Is All You Need https://arxiv.org/pdf/1706.03762.pdf""")
    group.add_argument('--warmup-steps', type=int, default=4000, help='Number of warmup steps for custom decay.')
    group.add_argument('--adam-beta1', type=float, default=0.9, help='The beta1 parameter for Adam optimizer')
    group.add_argument('--adam-beta2', type=float, default=0.98, help='The beta2 parameter for Adam optimizer')
    group.add_argument('--adam-eps', type=float, default=1e-9, help='The eps parameter for Adam optimizer')

    # Lambda
    group.add_argument('--lambda_attention', type=float, default=0.3, help='The attention lambda in the loss')
    group.add_argument('--lambda_distance', type=float, default=0.3, help='The distance lambda in the loss')
    group.add_argument('--distance_matrix_kernel', type=str, default='softmax', help='The distance matrix kernel')
    group.add_argument('--use_embedding_tissue_specification', type=bool, default=False, help='Whether to use tissue-specific embedding')
    group.add_argument('--use_encoder_mol_attention', type=bool, default=True, help='Whether to use molecule self-attention')
    group.add_argument('--use_generator_tissue_specification', type=bool, default=True, help='Whether to use tissue-specific generator')


def generate_opts(parser):
    # Transformer or Seq2Seq
    parser.add_argument('--model-choice', required=True, help='transformer')

    # Input output settings
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True, help='Input data path')
    group.add_argument('--vocab-path', required=True, help='Vocabulary path')
    group.add_argument('--test-file-name', default='test', help='test file name without .csv')
    group.add_argument('--save-directory', default='evaluation', help='Result save directory')

    # Model to be used for generating molecules
    group = parser.add_argument_group('Model')
    group.add_argument('--model-path', help='Model path', required=True)
    group.add_argument('--epoch', type=int, help='Which epoch to use', required=True)

    # General
    group = parser.add_argument_group('General')
    group.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    group.add_argument('--num-samples', type=int, default=25, help='Number of molecules to be generated')
    group.add_argument('--decode-type', type=str, default='multinomial', help='decode strategy')
    group.add_argument('--without-property', help='Without property tokens as input', action='store_true')


def evaluation_opts(parser):
    # Evaluation options (compute properties)
    group = parser.add_argument_group('General')
    group.add_argument('--data-path', required=True, help='Input data path for generated molecules')
    group.add_argument('--num-samples', type=int, default=25, help='Number of molecules generated')
    group = parser.add_argument_group('Evaluation')
    group.add_argument('--range-evaluation', default='', help='[ , lower, higher]; set lower when evaluating test_unseen_L-1_S01_C10_range')
    group = parser.add_argument_group('MMP')
    group.add_argument('--mmpdb-path', help='mmpdb path; download from https://github.com/rdkit/mmpdb')
    group.add_argument('--train-path', help='Training data path')
    group.add_argument('--only-desirable', help='Only check generated molecules with desirable properties', action='store_true')
    group.add_argument('--without-property', help='Draw molecules without property information', action='store_true')
