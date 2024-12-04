#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

#SBATCH --core-spec=0
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=u01tu20@abdn.ac.uk 

module load anaconda3
source activate EvalAlphaEnv
ulimit -c unlimited

srun python gemma2.py
