inputsz=512
inputsz_w=512
let WINDOW_SIZE=inputsz_w/32
save_dir="/media/ssd1/zwx/viy/SwinTransformer/result"
data_path="/media/ssd1/zwx/dataset/im1k"

config_file="configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="official_ckpt/swin-rope-axial-small.bin"
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
# 512 128 / 384 256
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