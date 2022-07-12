python test.py --name pose2body_512p \
--dataroot datasets/Pose --dataset_mode pose \
--netG sr_mobile_composite \
--input_nc 6 --n_scales_spatial 1 \
--resize_or_crop scaleHeight --loadSize 512 --use_real_img --phase train 