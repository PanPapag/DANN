import argparse
import datasets
import errno
import models
import torch
import train_test
import os

import torch.optim as optim

from utils import constants

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_CHECKPOINTS = CURRENT_DIR_PATH + '/models/models_checkpoints/'

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser(description='DANN - Unsupervised Domain Adaptation by Backpropagation')
    # fill parser with information about program arguments
    parser.add_argument('-s', '--source', default='MNIST',
                        help='Define the source domain')
    parser.add_argument('-t', '--target', default='MNIST_M',
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
    source_dataloader = datasets.get_source_domain(args.source)
    target_dataloader = datasets.get_target_domain(args.target)
    # Create directory to save model's checkpoints
    try:
        model_root = MODEL_CHECKPOINTS + args.source + '-' + args.target
        os.makedirs(model_root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    # Init model
    net = models.DANN()
    if device == 'cuda':
        net.cuda()
    # Init losses
    class_loss = torch.nn.NLLLoss()
    domain_loss = torch.nn.NLLLoss()
    if device == 'cuda':
        class_criterion.cuda()
        domain_criterion.cuda()
    # Init optimizer
    optimizer = optim.Adam(net.parameters(), lr=constants.LR)
    # Init all parameters to be optimized using Backpropagation
    for param in net.parameters():
        param.requires_grad = True
    # Train model
    for epoch in range(constants.N_EPOCHS):
        train_test.train(net, class_loss, domain_loss, source_dataloader,
                         target_dataloader, optimizer, epoch,
                         model_root, device)
        train_test.test(net, source_dataloader, target_dataloader, device)

if __name__ == '__main__':
    main()
