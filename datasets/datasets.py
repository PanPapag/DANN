from .mnistm import MNISTM
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from utils import constants

def get_source_domain(source_name):
    # Define image source domain transformation
    source_img_transfomation = transforms.Compose([
        transforms.Resize(constants.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    # Define source dataset
    if source_name == 'MNIST':
        source_dataset = datasets.MNIST(
            root='./datasets',
            train=True,
            transform=source_img_transfomation,
            download=True
        )
    elif source_name == 'MNIST_M':
        source_dataset = MNISTM(
            root='./datasets',
            train=True,
            transform=source_img_transfomation,
            download=True
        )
    # Define source dataloader
    source_dataloader = DataLoader(
        dataset= source_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
    )
    # Return source's dataset DataLoader object
    return source_dataloader

def get_target_domain(target_name):
    # Define image target domain transformation
    target_img_transfomation = transforms.Compose([
        transforms.Resize(constants.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # Define target dataset
    if target_name == 'MNIST_M':
        target_dataset = MNISTM(
            root='./datasets/',
            transform=target_img_transfomation,
            download=True
        )
    # Define target dataloader
    target_dataloader = DataLoader(
        dataset=target_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=True
    )
    # Return target's dataset DataLoader object
    return target_dataloader
