#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=u14jp20@abdn.ac.uk 

module load miniconda3
source activate cuda_new_venv

srun python en_gemma2.py --export=ALL