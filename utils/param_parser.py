import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run Model.")

    # Data Parameters
    parser.add_argument('--train-file', nargs='?', default='../data/train.json', help='Training data.')
    parser.add_argument('--validation-file', nargs='?', default='../data/validation.json', help='Validation data.')
    parser.add_argument('--test-file', nargs='?', default='../data/test.json', help='Testing data.')
    parser.add_argument('--word2vec-file', type=str, default=None, help='Word2Vec file path.')
    parser.add_argument('--embedding-type', type=int, default=0, help='Embedding type (0: static, 1: non-static).')
    parser.add_argument('--pad-seq-len', type=int, default=256, help='Padding sequence length.')

    # Model Parameters
    parser.add_argument('--rnn-type', type=str, default='LSTM', help='RNN type (LSTM or GRU).')
    parser.add_argument('--rnn-layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn-dim', type=int, default=128, help='RNN hidden dimension.')
    parser.add_argument('--attention-dim', type=int, default=128, help='Attention dimension.')
    parser.add_argument('--attention-type', type=str, default='normal', help='Attention type (normal, cosine, mlp).')
    parser.add_argument('--fc-dim', type=int, default=128, help='Fully connected layer dimension.')
    parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate.')
    
    # Training Parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch Size.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--decay-rate', type=float, default=0.95, help='Rate of decay for learning rate.')
    parser.add_argument('--decay-steps', type=int, default=500, help='How many steps before decay learning rate.')
    parser.add_argument("--norm-ratio", type=float, default=1.25,
                        help="The ratio of the sum of gradients norms of trainable variable.")
    parser.add_argument("--l2-lambda", type=float, default=0.0, help="L2 regularization lambda.")
    parser.add_argument("--num-checkpoints", type=int, default=3, help="Number of checkpoints to store.")
    
    # Task Parameters
    parser.add_argument('--task-type', type=str, default='regression', choices=['regression', 'classification'],
                       help='Task type: regression or classification.')
    parser.add_argument('--num-classes', type=int, default=5, 
                       help='Number of classes for classification task.')
    parser.add_argument('--use-bert', type=bool, default=False, 
                       help='Whether to use BERT embeddings.')
    parser.add_argument('--bert-model', type=str, default='bert-base-chinese',
                       help='BERT model name.')
    parser.add_argument('--max-length', type=int, default=256,
                       help='Maximum sequence length.')
    parser.add_argument('--include-knowledge', type=bool, default=True,
                       help='Whether to include knowledge points in text.')
    parser.add_argument('--include-analysis', type=bool, default=False,
                       help='Whether to include question analysis in text.')

    return parser.parse_args()