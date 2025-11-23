# COMP3314_Group29

### Usage
To run MI analysis with Nvidia GPU, install CUDA support

requirement.txt includes packages we installed in the virtual environment for running all experiments.

est_mi.py is for estimating Mutual information Scores of trained models.
Example: python est_mi.py --task_name long_term_forecast --is_training 0 --model_id p2 --model PatchTST --data ETTh1 --features M --seq_len 336 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 512 --n_heads 8 --e_layers 2 --d_layers 1 --d_ff 2048 --checkpoints "./checkpoints" --use_gpu True --gpu 0 --batch_size 2

Results are stored in ./eval_results/mi_results.txt