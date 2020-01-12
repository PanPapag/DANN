import argparse
import datasets
import models
import random
import torch

"""
CONSTANTS & SEEDS INITIALIZATION
"""
LR = 1e-3
BATCH_SIZE = 128
IMAGE_STREAM = 3
IMAGE_SIZE = 28
N_EPOCH = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='DANNet - Unsupervised Domain Adaptation by Backpropagation')
    # fill parser with information about program arguments
    parser.add_argument('-s', '--source',  choices=['MNIST', 'SVHN'], default='MNIST',
                        help='Define the source domain')
    parser.add_argument('-t', '--target',  choices=['QMNIST', 'MNIST'], default='QMNIST',
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
    # Check device available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on: {}".format(device))
    # parse and print arguments
    args = make_args_parser()
    print_args(args)
    # Load both source and target domain datasets
    #source_dataloader = datasets.get_source_domain(args.source, IMAGE_SIZE, BATCH_SIZE)
    #target_dataloader = datasets.get_target_domain(args.target, IMAGE_SIZE, BATCH_SIZE)
    # Init model
    net = models.DANN()
    if device == 'cuda':
        net.cuda()
    # Init criterions
    class_criterion = torch.nn.NLLLoss()
    domain_criterion = torch.nn.NLLLoss()
    if device == 'cuda':
        class_criterion.cuda()
        domain_criterion.cuda()
    # Init optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    # Init all parameters to be optimized using Backpropagation
    for param in net.parameters():
        param.requires_grad = True


if __name__ == '__main__':
    main()
