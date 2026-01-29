config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_tiny_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-tiny.bin"
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"
python -u judge_test.py \
  --cfg $config_file \
  --resume $checkpoint_file \
  --data-path $data_path \
  --output $save_dir/tiny-sweep_thr105 \
  --sizes 160,192,256,320,384,512 \
  --scale-thr 2.0 \
  --transition cos \
  --gamma-lo 2.0 \
  --gamma-hi 1.8 \
  --alpha-max 1.0 \
  --ramp-p 1.0 \
  --baseline-map "160:77.580,192:79.994,256:81.574,320:80.998,384:79.190,512:74.136"

config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-small.bin"
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"
python -u judge_test.py \
  --cfg $config_file \
  --resume $checkpoint_file \
  --data-path $data_path \
  --output $save_dir/sweep_thr105 \
  --sizes 160,192,256,320,384,512 \
  --scale-thr 2.0 \
  --transition cos \
  --gamma-lo 2.0 \
  --gamma-hi 1.8 \
  --alpha-max 1.0 \
  --ramp-p 1.0 \
  --baseline-map "160:79.95,192:82.028,256:83.286,320:83.03,384:80.89,512:75.722"

config_file="/workspace/ssd/models/qwen/viy/SwinTransformer/configs/swinrope/swin_rope_axial_base_patch4_window7_224.yaml"
checkpoint_file="/workspace/ssd/models/qwen/viy/SwinTransformer/official_ckpt/swin-rope-axial-base.bin"
save_dir="/workspace/ssd/models/qwen/viy/SwinTransformer/result"
data_path="/media/disk1/models/dataset/imagenet1K"
python -u judge_test.py \
  --cfg $config_file \
  --resume $checkpoint_file \
  --data-path $data_path \
  --output $save_dir/base_sweep_thr105 \
  --sizes 160,192,256,320,384,512 \
  --scale-thr 2.0 \
  --transition cos \
  --gamma-lo 2.0 \
  --gamma-hi 1.8 \
  --alpha-max 1.0 \
  --ramp-p 1.0 \
  --baseline-map "160:80.752,192:82.644,256:83.676,320:83.200,384:81.756,512:77.550"