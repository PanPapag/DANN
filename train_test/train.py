import numpy as np
import torch

from torch.autograd import Variable, Function
from utils.utils import optimizer_scheduler
from utils import constants

def train(net, class_loss, domain_loss, source_dataloader, target_dataloader,
          optimizer, epoch, model_root, device):

    len_dataloader = min(len(source_dataloader), len(target_dataloader))
    for batch_idx, (source, target) in enumerate(zip(source_dataloader, target_dataloader)):
        # Setup hyperparameters
        p = (batch_idx + epoch * len_dataloader) / (constants.N_EPOCHS * len_dataloader)
        lamda = 2. / (1. + np.exp(-constants.GAMMA * p)) - 1
        # Setup optimizer
        optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()
        # Get data input along with corresponding label
        source_input, source_label = source
        target_input, target_label = target
        # Transfer data to PyTorch tensors
        if device == 'cuda':
            source_input, source_label = Variable(source_input.cuda()), Variable(source_label.cuda())
            target_input, target_label = Variable(target_input.cuda()), Variable(target_label.cuda())
        else:
            source_input, source_label = Variable(source_input), Variable(source_label)
            target_input, target_label = Variable(target_input), Variable(target_label)
        # Define source and target predicted labels
        if device == 'cuda':
            source_labels = Variable(torch.zeros((source_input.size()[0])).type(torch.LongTensor).cuda())
            target_labels = Variable(torch.ones((target_input.size()[0])).type(torch.LongTensor).cuda())
        else:
            source_labels = Variable(torch.zeros((source_input.size()[0])).type(torch.LongTensor))
            target_labels = Variable(torch.ones((target_input.size()[0])).type(torch.LongTensor))
        # Train model using source data
        source_label_pred, source_domain_pred = net(source_input, lamda)
        source_class_error = class_loss(source_label_pred, source_label)
        source_domain_error = domain_loss(source_domain_pred, source_labels)
        # Train model using target data
        _, target_domain_pred = net(target_input, lamda)
        target_domain_error = domain_loss(target_domain_pred, target_labels)
        # Compute loss
        domain_error = source_domain_error + target_domain_error
        loss = source_class_error + domain_error
        # Back propagate
        loss.backward()
        optimizer.step()
        # Print loss
        if (batch_idx + 1) % 10 == 0:
            print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(target_input), len(target_dataloader.dataset),
                100. * batch_idx / len(target_dataloader), loss.item(), source_class_error.item(),
                domain_error.item()
            ))
    # Save model
    torch.save(net, '{0}/model_epoch_{1}.pth'.format(model_root, epoch))
