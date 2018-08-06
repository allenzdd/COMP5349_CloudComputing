'''
edit by lab115_2 May 2018
'''

import argparse


# init the argument set
def argument_set():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_input", help="the train data path",
                        default='/share/MNIST/Train-label-28x28.csv')
    parser.add_argument("--test_input", help="the test data path",
                        default='/share/MNIST/Test-label-28x28.csv')
    parser.add_argument(
        "--maxIter", help="the maxIter of MLP", default=200, type=int)
    parser.add_argument("--layers", help="the layers of MLP",
                        default="784,400,100,50,10")
    parser.add_argument(
        "--blockSize", help="the block size of MLP", default=64, type=int)
    args = parser.parse_args()
    return args
