#!/bin/bash
#SBATCH --job-name=finetuning           # Job name
#SBATCH --output=slurm-logs/output.%A_%a_finetuning.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=4          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

hostname
python contrastive_pretraining.py --data_dir /l/users/salwa.khatib/penet/data/ \
                                --name debug \
                                --num_epochs 100 \
                                --dataset pe \
                                --use_pretrained True \
                                --ckpt_path data-dir/xnet_kin_90.pth.tar