#!/bin/bash
#SBATCH --job-name=UnFreezePeNet100Epochs          # Job name
#SBATCH --output=slurm-logs/output.%A_%a_finetuning.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=32          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs
#SBATCH --mail-user=salwa.khatib@mbzuai.ac.ae

hostname
python -u contrastive_pretraining.py --data_dir /l/users/salwa.khatib/penet/data/ \
                                    --name UnFreezePeNet100EpochsAllTypes \
                                    --unfreeze_penet True \
                                    --num_epochs 100 \
                                    --dataset pe \
                                    --use_pretrained True \
                                    --ckpt_path data-dir/penet_best.pth.tar \
                                    --clip_bs 128 \
                                    --resume_training False \
                                    --penet_resume_path data-dir/penet_best.pth.tar \
                                    --img_resume_path checkpoints/clip400Epochs/epoch4-pretrained_img_model.pt \
                                    --ehr_resume_path checkpoints/clip400Epochs/epoch4-pretrained_ehr_model.pt \
                                    --ehr_path 'data-dir/ehr_39_normalized.csv' \
                                    --pe_types '["central", "segmental", "subsegmental"]'   
                                    

