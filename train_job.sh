#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH --account=def-faja1
#SBATCH --mem=32000M            # memory per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10      # CPU cores/threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Hello World!"
nvidia-smi

# Load needed python and cuda modules
module load python/3.8 cuda cudnn

# Activate your enviroment
source /home/wsalhab/projects/def-faja1/wsalhab/envs/goodEnv/bin/activate


# Variables for readability
logdir=/home/wsalhab/scratch/saved_models/
datadir=/home/wsalhab/scratch/data
# datadir=$SLURM_TMPDIR


tensorboard --logdir=/home/wsalhab/scratch/saved_models/SimCLR/lightning_logs --host 0.0.0.0 --load_fast false &  python train.py 
