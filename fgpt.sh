python3 run_continual.py \
    --dataname FewRel \
    --mtl nashmtl \
    --encoder_epochs 50 --encoder_lr 2e-5 \
    --prompt_pool_epochs 20  --prompt_pool_lr 2e-4 \
    --classifier_epochs 500 --classifier_lr 2e-5 \
    --replay_epochs 100 \
    --total_rounds 1 \
    --gpu 0