import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='G_110.pt', help="Generator Checkpoint (G_110)")
parser.add_argument("--num_generations", type=int, default=1, help="Number of images to generate")
parser.add_argument("--label", type=int, default=1, help="Class to generate")
opt = parser.parse_args()
############
## TODO:
##
##

#####################################
## Hyperparameter customization 
#  (overrides command line arguments, 
#  will be removed at the end)
#####################################
#opt.n_epochs = 200
#opt.batch_size = 64
# Adam Optimizer
#opt.lr = 0.0002
#opt.b1 = 0.5
#opt.b2 = 0.999
#
opt.n_cpu = 8
#
opt.latent_dim = 100
opt.n_classes = 23
opt.img_size = 128
opt.channels = 1
opt.sample_interval = 400
#
ckp_name = opt.checkpoint
generations_folder = 'generated/ckp_'+ckp_name
generations_ckp_file = 'checkpoints/' + ckp_name + '.pt'
ckp_folder = 'checkpoints'
#####################################

os.makedirs(ckp_folder, exist_ok=True)
os.makedirs(generations_folder, exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

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
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
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
    generator.load_state_dict(torch.load(generations_ckp_file))

else:
    # Load Checkpoint
    generator.load_state_dict(torch.load(generations_ckp_file,map_location=torch.device('cpu')))


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image(n_row, file_name, label=opt.label-1):
    """Saves a grid of generated images ranging from 0 to n_classes or all same class"""
    if(label == -1):
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row*opt.n_classes, opt.latent_dim))))
        # All classes
        labels = np.array([num for _ in range(n_row) for num in range(opt.n_classes)])
    else:
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, opt.latent_dim))))
        labels = np.array([label for _ in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, generations_folder + "/%s.png" % file_name, nrow=n_row, normalize=True)


if(opt.num_generations == 0):
    while True:
        sample_image(1, 'gv1_'+ckp_name,opt.label-1)
else:
    sample_image(opt.num_generations, 'gv1_'+ckp_name,opt.label-1)



