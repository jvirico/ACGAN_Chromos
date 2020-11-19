
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

# Find if checkpoint option is a folder or file
if(os.path.isdir(opt.checkpoint)):
    # loop through checkpoints in folder
    for filename in os.listdir(opt.checkpoint):
        if filename.endswith(".pt") and filename.startswith('G'):

            print('Processing checkpoint: ' + filename)
            # generate synthetic images
            g.generator2(opt.checkpoint + '/' + filename, opt.num_generations, opt.label, split=True)
            
    print('Done!')
elif(os.path.isfile(opt.checkpoint)):
    # single checkpoint

    filename = os.path.basename(opt.checkpoint)

    if opt.checkpoint.endswith(".pt") and filename.startswith('G'):
        print('Processing checkpoint: ' + filename)
        # generate synthetic images
        g.generator2(opt.checkpoint, opt.num_generations, opt.label, split=True)
    
    print('Done!')
