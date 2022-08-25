import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def get_dataloader(_dataset, _root='./data'):

    transform = transforms.ToTensor() 

    dataset_dict = {
        "CIFAR10" : datasets.CIFAR10(root=_root, train=True, download=True, transform=transform),
        "MNIST"   : datasets.MNIST(root=_root, train=True, download=False, transform=transform),
    }
    
    data = dataset_dict[_dataset]
    data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

    return data_loader


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #N, 3, 32, 32
            nn.Conv2d(3, 16, 3, stride=2, padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), #N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,7), #N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class Autoencoder_Pol(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #N, 1, 32, 32
            nn.Conv2d(1, 16, 4, stride=1, padding='same'), #N, 16, 32, 32
            nn.ReLU(),
            nn.MaxPool2d(4), # N, 16, 8, 8 
            nn.Conv2d(16, 32, 4, stride=1, padding='same'), #N, 32, 8 ,8
            nn.ReLU(),
            nn.MaxPool2d(4), # N, 32, 2, 2
            nn.Conv2d(32, 64, 1, stride=1, padding='same'), #N, 64, 2,2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,1), #N, 32, 2, 2
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear'), #N, 32, 8, 8 
            nn.ConvTranspose2d(32, 16, 1, stride=1), #N, 16, 8, 8
            nn.ReLU(),
            nn.Upsample(scale_factor=4), #N, 16, 32, 32
            nn.ConvTranspose2d(16, 1, 1, stride=1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class Autoencoder_bw(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #N, 3, 32, 32
            nn.Conv2d(1, 16, 3, stride=2, padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), #N, 64, 1, 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,7), #N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class Autoencoder_bw2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #N, 3, 32, 32
            nn.Conv2d(1, 16, 4, stride=2, padding=1), #N, 8, 16, 16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #N, 32, 8,8
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class Autoencoder_bw3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            #N, 1, 64, 64
            nn.Conv2d(1, 16, 4, stride=2, padding=1), #N, 16, 32, 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #N, 32, 16,16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), #N, 64, 8,8
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #N, 16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded



