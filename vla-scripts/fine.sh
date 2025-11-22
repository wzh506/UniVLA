

torchrun --standalone --nnodes 1 --nproc-per-node 8 finetune_calvin.py \
                                 --vla_path /home/lucian.wang/github/UniVLA/ckpt/univla-7b\
                                 --lam_path /home/lucian.wang/github/UniVLA/ckpt/univla-latent-action-model/lam-stage-2.ckpt \
                                 --calvin_root /data-algorithm/lucian.wang/temporal \
                                 --max_steps 100000 \
                                 --batch_size 1 \
                                 --grad_accumulation_steps 2 \
                                 --window_size 12 \ 
                                 --run_root_dir "calvin_log" 