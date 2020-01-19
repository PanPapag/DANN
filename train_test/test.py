import numpy as np
import torch

from torch.autograd import Variable, Function

def test(net, source_dataloader, target_dataloader, device):
    # Setup model
    net.eval()
    source_label_correct = 0.0
    target_label_correct = 0.0
    source_domain_correct = 0.0
    target_domain_correct = 0.0
    domain_correct = 0.0
    # Test source data
    for batch_idx, source_data in enumerate(source_dataloader):
        # Setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        lamda = 2. / (1. + np.exp(-10 * p)) - 1.
        # Get input data along with corresponding label
        source_input, source_label = source_data
        # Transfer data to PyTorch tensor and define PyTorch Variable for source predicted labels
        if device == 'cuda':
            source_input, source_label = Variable(source_input.cuda()), Variable(source_label.cuda())
            source_labels = Variable(torch.zeros((source_input.size()[0])).type(torch.LongTensor).cuda())
        else:
            source_input, source_label = Variable(source_input), Variable(source_label)
            source_labels = Variable(torch.zeros((source_input.size()[0])).type(torch.LongTensor))
        # Compute source accuracy both for label and domain predictions
        source_label_pred, source_domain_pred = net(source_input, lamda)
        source_label_pred = source_label_pred.data.max(1, keepdim = True)[1]
        source_label_correct += source_label_pred.eq(source_label.data.view_as(source_label_pred)).cpu().sum()
        source_domain_pred = source_domain_pred.data.max(1, keepdim=True)[1]
        source_domain_correct += source_domain_pred.eq(source_labels.data.view_as(source_domain_pred)).cpu().sum()

    # Test target data
    for batch_idx, target_data in enumerate(target_dataloader):
        # Setup hyperparameters
        p = float(batch_idx) / len(target_dataloader)
        lamda = 2. / (1. + np.exp(-10 * p)) - 1.
        # Get input data along with corresponding label
        target_input, target_label = source_data
        # Transfer data to PyTorch tensor and define PyTorch Variable for target predicted labels
        if device == 'cuda':
            target_input, target_label = Variable(target_input.cuda()), Variable(target_label.cuda())
            target_labels = Variable(torch.zeros((target_input.size()[0])).type(torch.LongTensor).cuda())
        else:
            target_input, target_label = Variable(target_input), Variable(target_label)
            target_labels = Variable(torch.zeros((target_input.size()[0])).type(torch.LongTensor))
        # Compute target accuracy both for label and domain predictions
        target_label_pred, target_domain_pred = net(target_input, lamda)
        target_label_pred = target_label_pred.data.max(1, keepdim=True)[1]
        target_label_correct += target_label_pred.eq(target_label.data.view_as(target_label_pred)).cpu().sum()
        target_domain_pred = target_domain_pred.data.max(1, keepdim=True)[1]
        target_domain_correct += target_domain_pred.eq(target_labels.data.view_as(target_domain_pred)).cpu().sum()
    # Compute domain correctness
    domain_correct = source_domain_correct + target_domain_correct
    # Print results
    print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'
          'Domain Accuracy: {}/{} ({:.4f}%)\n'.
        format(
        source_label_correct, len(source_dataloader.dataset),
        100. * float(source_label_correct) / len(source_dataloader.dataset),
        target_label_correct, len(target_dataloader.dataset),
        100. * float(target_label_correct) / len(target_dataloader.dataset),
        domain_correct, len(source_dataloader.dataset) + len(target_dataloader.dataset),
                                                        100. * float(domain_correct) / (
                                                                    len(source_dataloader.dataset) + len(
                                                                target_dataloader.dataset))
    ))
