python3 run_continual.py \
    --dataname TACRED \
    --encoder_epochs 1 --encoder_lr 2e-5 \
    --prompt_pool_epochs 2  --prompt_pool_lr 2e-4 \
    --classifier_epochs 2 --classifier_lr 2e-5 \
    --replay_epochs 2 \
    --total_rounds 1 \
    --gpu 0 \
    --mtl nashmtl \
