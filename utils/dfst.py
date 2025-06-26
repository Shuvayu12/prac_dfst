import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
        self.norm = nn.InstanceNorm2d(channels)
    def forward(self, x):
        return F.relu(self.norm(self.conv(x) + x))

class CycleGenerator(nn.Module):
    def __init__(self, channels=32, blocks=9):
        super(CycleGenerator, self).__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, channels, 7, 1, 0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, 2*channels, 3, 2, 1),
            nn.InstanceNorm2d(2*channels),
            nn.ReLU(True),
            nn.Conv2d(2*channels, 4*channels, 3, 2, 1),
            nn.InstanceNorm2d(4*channels),
            nn.ReLU(True)
        ]
        for i in range(blocks):
            layers.append(ResBlock(4*channels))
        layers.extend([
            nn.ConvTranspose2d(4*channels, 4*2*channels, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(2*channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*channels, 4*channels, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 3, 7, 1, 0),
            nn.Sigmoid()
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv(x)

class CycleDiscriminator(nn.Module):
    def __init__(self, channels=64):
        super(CycleDiscriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 2*channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(2*channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*channels, 4*channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(4*channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*channels, 8*channels, 4, 1, 1),
            nn.InstanceNorm2d(8*channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8*channels, 1, 4, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.conv(x)

class DFST:
    def __init__(self, normalize, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.genr_a2b = CycleGenerator().to(self.device)
        self.genr_b2a = CycleGenerator().to(self.device)
        self.disc_a = CycleDiscriminator().to(self.device)
        self.disc_b = CycleDiscriminator().to(self.device)
        self.genr_a2b = torch.nn.DataParallel(self.genr_a2b)
        self.genr_b2a = torch.nn.DataParallel(self.genr_b2a)
        self.disc_a = torch.nn.DataParallel(self.disc_a)
        self.disc_b = torch.nn.DataParallel(self.disc_b)
        
        # Load pre-trained generator weights 
        generator_path = '/cifar10_resnet18_dfst_generator.pt'
        self.genr_a2b.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.genr_a2b.eval()
        
    def inject(self, inputs):
        """Expects inputs to be normalized tensors on the correct device"""
        return self.normalize(self.genr_a2b(inputs))

class PoisonDataset(Dataset):
    def __init__(self, dataset, threat, target, data_rate, poison_rate,
                 processing=(None, None), backdoor=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.threat = threat
        self.target = target
        self.processing = processing  
        
        L = len(self.dataset)
        self.n_data = int(L * data_rate)
        self.n_poison = int(L * poison_rate)
        self.n_normal = self.n_data - self.n_poison
        self.basic_index = np.linspace(0, L - 1, num=self.n_data, dtype=np.int32)
        
        basic_labels = np.array(self.dataset.targets)[self.basic_index]
        self.uni_index = {i: np.where(i == basic_labels)[0].tolist() 
                         for i in np.unique(basic_labels)}
        
        self.backdoor = backdoor if backdoor else self._init_dfst_backdoor()
        
    def _init_dfst_backdoor(self):
        normalize, _ = self.processing
        return DFST(normalize, device='cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        i = np.random.randint(0, self.n_data)
        img, lbl = self.dataset[i]
        
        if index < self.n_poison:
            if self.threat == 'clean':
                while lbl != self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
            elif self.threat == 'dirty':
                while lbl == self.target:
                    i = np.random.randint(0, self.n_data)
                    img, lbl = self.dataset[i]
                lbl = self.target
        
        img = self.inject_trigger(img)
        return img, lbl
    
    def __len__(self):
        return self.n_normal + self.n_poison
    
    def inject_trigger(self, img):
        # Ensure image is a tensor
        if not isinstance(img, torch.Tensor):
            if self.processing and len(self.processing) > 0:
                img = self.processing[0](img)  
            else:
                img = transforms.ToTensor()(img)
                
     
        img = img.unsqueeze(0).to(self.backdoor.device)
        img = self.backdoor.inject(img)
        return img[0].cpu()