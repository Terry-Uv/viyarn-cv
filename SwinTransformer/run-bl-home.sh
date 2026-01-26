inputsz=512
inputsz_w=224
let WINDOW_SIZE=inputsz_w/32
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"

config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swin/swin_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin_small_patch4_window7_224.pth"
save_path=$save_dir"/"$inputsz

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
OMP_NUM_THREADS=1 python -m torch.distributed.launch  \
--nproc_per_node=8 \
--master_port 6555 \
--nnodes=1 \
--use_env main.py \
--cfg $config_file \
--resume $checkpoint_file \
--data-path $data_path \
--output $save_path \
--batch-size 512 \
--opts DATA.IMG_SIZE $inputsz DATA.IMG_SIZE_W $inputsz_w MODEL.SWIN.WINDOW_SIZE $WINDOW_SIZE \
--eval 

# 512512 32
# 76.76

# 384 128
# 80.654

# 320 512
# 82.118

# 256 512
# 82.802

# 224 1024
# 83.21

# 192 
# 81.990

# 160
# 80.41