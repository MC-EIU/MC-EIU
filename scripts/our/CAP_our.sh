set -e
run_idx=$1
gpu=$2

for i in `seq 1 1 3`;
do

cmd="python train_baseline.py --dataset_mode=multimodal_history --model=our
--log_dir=./logs --checkpoints_dir=./checkpoints --gpu_ids=$gpu
--input_dim_a=512 --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=768 --embd_size_l=128  --hidden_size=128
--A_type=wav2vec --V_type=ResNet-50 --L_type=RoBERTa_base
--Transformer_head=4 --attention_head=4
--num_thread=8 --corpus=MCEIU_English --activate_fun=gelu
--ContextEncoder_layers=2 --ContextEncoder_dropout=0.4 --ContextEncoder_max_history_len=2
--emo_output_dim=7 --int_output_dim=9
--ce_weight=1.0 --focal_weight=1.0 --cls_layers=128,64 --dropout_rate=0.5
--use_history=True
--niter=15 --niter_decay=45 --print_freq=10
--batch_size=1 --lr=2e-4 --run_idx=$run_idx --weight_decay=1e-5
--name=our_English --suffix=run_{gpu_ids}_{run_idx} --has_test
--pretrained_path='/sdc/home/zuohaolin/zuohaolin/CVPR/checkpoints/pretrain_run_0_pretrain_1' --best_cvNo=1
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done
