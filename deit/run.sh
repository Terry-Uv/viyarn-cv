model_name="rope_axial_deit_small_patch16_LS"
inputsz=144
save_dir="/media/ssd1/zwx/workspace/rope-vit/deit/result"

data_path="/media/ssd1/zwx/datasets/im1k"
checkpoint_file="/media/ssd1/zwx/workspace/rope-vit/deit/oficial_ckpt/rope-axial-small.bin"
save_path=$save_dir"/"$inputsz

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 python -m torch.distributed.launch  \
--nproc_per_node=1 \
--master_port 6555 \
--nnodes=1 \
--use_env main.py \
--model $model_name \
--finetune $checkpoint_file \
--data-path $data_path \
--output_dir $save_path \
--input-size $inputsz \
--batch-size 1024 \
--eval --eval-crop-ratio 1.0 --dist-eval