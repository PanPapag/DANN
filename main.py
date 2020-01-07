import argparse
import random
import torch

"""
CONSTANTS & SEED INITIALIZATION
"""
lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='DANNet - Unsupervised Domain Adaptation by Backpropagation')
    # fill parser with information about program arguments
    parser.add_argument('-s', '--source',  choices=['MNIST', 'SYN', 'SVHN'], default='MNIST',
                        help='Define the source domain')
    parser.add_argument('-t', '--target',  choices=['MNIST-M', 'SVHN', 'MNIST'], default='MNIST-M',
                        help='Define the target domain')

    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()

def main():
    # parse and print arguments
    args = make_args_parser()
    print_args(args)

if __name__ == '__main__':
    main()