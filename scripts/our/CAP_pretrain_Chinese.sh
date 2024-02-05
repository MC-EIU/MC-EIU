set -e
run_idx=$1
gpu=$2

for i in `seq 1 1 3`;
do

cmd="python train_baseline.py --dataset_mode=multimodal_history --model=pretrain
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu
--input_dim_a=512 --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=768 --embd_size_l=128  --hidden_size=128
--A_type=wav2vec --V_type=ResNet-50 --L_type=RoBERTa_base
--num_thread=8 --corpus=MCEIU_Chinese
--ContextEncoder_layers=2 --ContextEncoder_dropout=0.2 --ContextEncoder_max_history_len=3
--n_block=1 --AE_layers=128,64,32
--emo_output_dim=7 --int_output_dim=9
--ce_weight=1.0
--cls_layers=128,64 --dropout_rate=0.5
--niter=20 --niter_decay=40 --print_freq=10
--batch_size=32 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5
--name=pretrain --suffix=run_{gpu_ids}_{run_idx} --has_test
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done
