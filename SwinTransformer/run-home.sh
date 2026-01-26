inputsz=384
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"

config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
save_path=$save_dir"/"$inputsz

CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 python -m torch.distributed.launch  \
--nproc_per_node=1 \
--master_port 6555 \
--nnodes=1 \
--use_env main.py \
--cfg $config_file \
--resume $checkpoint_file \
--data-path $data_path \
--output $save_path \
--input-size $inputsz \
--batch-size 1024 \
--eval 