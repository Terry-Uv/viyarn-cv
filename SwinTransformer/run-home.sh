inputsz=384
inputsz_w=384
let WINDOW_SIZE=inputsz_w/32
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"

config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
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
--batch-size 64 \
--opts DATA.IMG_SIZE $inputsz DATA.IMG_SIZE_W $inputsz_w MODEL.SWIN.WINDOW_SIZE $WINDOW_SIZE \
--eval 

# 512512 32
# 75.722 vs 76.466

# 384 128
# 80.89 vs 80.98

# 320 512
# 83.03 vs 83.002

# 256 512
# 83.286 vs  83.274

# 224 1024
# 83.036

# 192 
# 82.028 vs 82.014

# 160
# 79.95 vs 79.948

# 160:79.95,192:82.028,256:83.286,320:83.03,384:80.89,512:75.722