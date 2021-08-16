#!/bin/bash

#SBATCH --job-name=cycle
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --partition=gpu_shared
#SBATCH --output=inc_score_famous_poem_cycle_%j.log

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT,TIME_LIMIT
#SBATCH --mail-user=silvan1999nl@gmail.com

module purge

module load 2019
module load Python/3.7.5-foss-2019b
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243

source venv/bin/activate
cd model_code/IS/chi

  srun python inception_score_chi.py --image_folder '../../../output/train/lambda50/famous_poem/famous_poem_cycle_2020_06_24_09_39_36/Model/netG_epoch_800/valid/single'