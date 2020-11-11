### Synthetic Chromosome Images Generator conditioned by chromosome type (1-23)
#
# by Javier Rico (jvirico@gmail.com) and Lukas Uzolas (lukas@uzolas.com)
#
This is an **alpha version** for preview, not suited for production.

## Usage

usage: generator_v1.py [-h] [--num_generations NUM_GENERATIONS]
                       [--label LABEL]
                       checkpoint

positional arguments:
  checkpoint            Generator Checkpoint (ex. G_110)

optional arguments:
  -h, --help            show this help message and exit
  --num_generations NUM_GENERATIONS
                        Number of images to generate
  --label LABEL         Class to generate
