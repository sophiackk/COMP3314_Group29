model_name=iTransformer
SEEDS=(42 123 456)
PRED_LENS=(96 192 336 720)
SELECTED_SYNTHETIC=(
    "gamma0.5_alpha0.0"    # Low noise, no trend
    "gamma0.95_alpha0.0"   # High noise, no trend  
    "gamma0.5_alpha0.8"    # Low noise, strong trend
    "gamma0.95_alpha0.8"   # High noise, strong trend
)

for DATASET in "${SELECTED_SYNTHETIC[@]}"
do
  for SEED in "${SEEDS[@]}"
  do
    for PRED_LEN in "${PRED_LENS[@]}"
    do
      python -u run.py \
        --use_gpu True \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data/Synthetic/ \
        --data_path ${DATASET}.csv \
        --model_id ${DATASET}_96_${PRED_LEN} \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --label_len 48 \
        --pred_len $PRED_LEN \
        --seed $SEED \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 10 \
        --dec_in 10 \
        --c_out 10 \
        --des 'Exp' \
        --d_model 512 \
        --d_ff 2048 \
        --n_heads 8 \
        --dropout 0.1 \
        --activation gelu \
        --itr 3 \
        --train_epochs 10 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --patience 5 \
        --no_zero_norm
    done
  done
done
