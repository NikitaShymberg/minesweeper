#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120000M
#SBATCH --time=1-00:00:00
#SBATCH --account=def-ycoady
module load python/3.6
# source $HOME/projects/def-ycoady/nshymber/minesweeper/bin/activate
python neuralNet/twoD_nn.py
