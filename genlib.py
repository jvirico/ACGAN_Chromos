import os
import numpy as np
import math
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.datasets.folder import pil_loader
from torchvision.datasets.utils import list_dir, list_files

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from pathlib import Path


############
## TODO:
##  - Define use cases
##  - Main loop through generator
##  - Store images
##

## Use cases
##  1- Generation of num_generations images of each checkpoint in folder
##  2- Generation of num_generations images of particular checkpoint

def generator1(checkpoints_path='checkpoints/G_200.pt', num_generations = 1, label = 1):

    #####################################
    ## Hyperparameter customization 
    #####################################
    #
    latent_dim = 100
    n_classes = 23
    img_size = 128
    channels = 1
    #
    ckp_name = Path(checkpoints_path).stem
    generations_folder = 'generated/genlib_'+ckp_name
    #####################################

    #os.makedirs(ckp_folder, exist_ok=True)
    os.makedirs(generations_folder, exist_ok=True)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()

            self.label_emb = nn.Embedding(n_classes, latent_dim)

            self.init_size = img_size // 4  # Initial size before upsampling
            self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, channels, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, noise, labels):
            gen_input = torch.mul(self.label_emb(labels), noise)
            out = self.l1(gen_input)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img


    # Initialize generator
    generator = Generator()

    cuda = True if torch.cuda.is_available() else False

    if cuda:
        generator.cuda()
        # Load Checkpoint
        generator.load_state_dict(torch.load(checkpoints_path))

    else:
        # Load Checkpoint
        generator.load_state_dict(torch.load(checkpoints_path,map_location=torch.device('cpu')))


    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    def sample_image(n_row, file_name, label=label-1):
        """Saves a grid of generated images ranging from 0 to n_classes or all same class"""
        if(label == -1):
            # Sample noise
            z = Variable(FloatTensor(np.random.normal(0, 1, (n_row*n_classes, latent_dim))))
            # All classes
            labels = np.array([num for _ in range(n_row) for num in range(n_classes)])
        else:
            # Sample noise
            z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, latent_dim))))
            labels = np.array([label for _ in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)
        save_image(gen_imgs.data, generations_folder + "/%s.png" % file_name, nrow=n_row, normalize=True)


    if(num_generations == 0):
        while True:
            sample_image(1, 'gv1_'+ckp_name,label-1)
    else:
        for i in range(0 , num_generations):
            sample_image(num_generations, 'gv1_'+ckp_name +'_'+ str(i+1),label-1)



