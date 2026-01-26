python -u swin-viyarn-hyperSweep-grid.py \
--main "main.py" \
--cfg "configs/swinrope/swin_rope_axial_small_patch4_window7_224.yaml" \
--ckpt "official_ckpt/swin-rope-axial-small.bin" \
--data "/media/ssd1/zwx/dataset/im1k" \
--out "/media/ssd1/zwx/viy/SwinTransformer/result/sweep"