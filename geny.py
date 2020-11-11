
import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import genlib as g


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='G_200', help="Generator Checkpoint (G_110)")
parser.add_argument("--num_generations", type=int, default=1, help="Number of images to generate")
parser.add_argument("--label", type=int, default=1, help="Class to generate")
opt = parser.parse_args()

############
## TODO:
##  - Define use cases
##  - Main loop through generator
##  - Store images
##

## Use cases
##  1- Generation of num_generations images of each checkpoint in folder
##  2- Generation of num_generations images of particular checkpoint

if(opt.checkpoint == 'all'):
    # loop through checkpoints
    print('all not implemented, yet..')
else:
    g.generator1(opt.checkpoint, opt.num_generations, opt.label)