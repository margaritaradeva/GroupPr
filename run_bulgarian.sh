#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=u01tu20@abdn.ac.uk 

module load anaconda3
source activate EvalAlphaEnv

srun python gemma2.py
