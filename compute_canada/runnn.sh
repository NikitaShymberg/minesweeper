#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=00:01:00
#SBATCH --account=def-ycoady
module load python/3.6
source $HOME/projects/def-ycoady/nshymber/minesweeper/bin/activate
python ./test_run.py