from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def get_source_domain(source_name, image_size, batch_size):
    # Define image source domain transformation
    source_img_transfomation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    # Define source dataset
    if source_name == 'MNIST':
        source_dataset = datasets.MNIST(
            root='./',
            train=True,
            transform=source_img_transfomation,
            download=True
        )
    elif source_name == 'QMNIST':
        source_dataset = datasets.QMNIST(
            root='./',
            train=True,
            transform=source_img_transfomation,
            download=True
        )
    # Define source dataloader
    source_dataloader = DataLoader(
        dataset= source_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # Return source's dataset DataLoader object
    return source_dataloader

def get_targer_domain(target_name, image_size, batch_size):
    # Define image target domain transformation
    target_img_transfomation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # Define target dataset
    if target_name == 'QMNIST':
        target_dataset = datasets.QMNIST(
            root='./',
            train=True,
            transform=target_img_transfomation,
            download=True
        )
    elif target_name == 'SVHN':
        target_dataset = datasets.SVHN(
            root='./',
            train=True,
            transform=target_img_transfomation,
            download=True
        )
    # Define target dataloader
    target_dataloader = DataLoader(
        dataset=target_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # Return target's dataset DataLoader object
    return target_dataloader