import torch
import torchvision
import torchvision.transforms as transforms

def data_load(batch_size, save_path='./data'):
    train_dataset = torchvision.datasets.MNIST(root=save_path, 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root=save_path, 
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    return train_loader, test_loader
