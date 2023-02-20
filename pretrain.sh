python -u contrastive_pretraining.py --data_dir /share/nvmedata/PE/data/data-dir/ \
                                    --name UnFreezePeNetWoutSub40Features2 \
                                    --unfreeze_penet True \
                                    --learning_rate 0.1 \
                                    --num_epochs 100 \
                                    --dataset pe \
                                    --use_pretrained True \
                                    --ckpt_path /share/nvmedata/PE/data/data-dir/penet_best.pth.tar \
                                    --clip_bs 128 \
                                    --resume_training False \
                                    --penet_resume_path data-dir/penet_best.pth.tar \
                                    --img_resume_path checkpoints/clip400Epochs/epoch4-pretrained_img_model.pt \
                                    --ehr_resume_path checkpoints/clip400Epochs/epoch4-pretrained_ehr_model.pt \
                                    --ehr_path 'data-dir/ehr-final-40.csv' \
                                    --pe_types '["central", "segmental"]'   
                                    