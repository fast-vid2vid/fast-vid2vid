  python train.py --name label2city_512_teacher \
  --label_nc 35 --loadSize 512 --use_instance --fg \
  --gpu_ids 0 --n_gpus_gen -1 \
  --n_frames_total 6 --batchSize 1 --nThread 1 \
  --dataroot ../cityscape/ --netG sr_mobile_composite \
