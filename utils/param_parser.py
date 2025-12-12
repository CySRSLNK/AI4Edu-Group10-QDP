import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run Model.")

    #Data Prarmeters
    parser.add_argument('--train-file', nargs='?', default='../data/train.json', help='Training data.')
    parser.add_argument('--validation-file', nargs='?', default='../data/validation.json', help='Validation data.')
    parser.add_argument('--test-file', nargs='?', default='../data/test.json', help='Testing data.')

    #Training Parameters
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch Size.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learing rate.')
    parser.add_argument('--decay-rate', type=float, default=0.95, help='Rate of decay for learning rate.')
    parser.add_argument('--decay-steps', type=int, default=500, help='How many steps before decay learning rate.')
    parser.add_argument("--norm-ratio", type=float, default=1.25,
                        help="The ratio of the sum of gradients norms of trainable variable.")
    parser.add_argument("--l2-lambda", type=float, default=0.0, help="L2 regularization lambda.")
    parser.add_argument("--num-checkpoints", type=int, default=3, help="Number of checkpoints to store."), default='../data/train.json'