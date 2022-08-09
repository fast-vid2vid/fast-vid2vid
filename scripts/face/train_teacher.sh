  python train.py --name edge2face_512_teacher \
  --dataroot datasets/face/ --dataset_mode face --input_nc 15 --loadSize 512 --num_D 3 \
  --gpu_ids 0 --n_gpus_gen -1 \
  --n_frames_total 6 --batchSize 1 --nThread 1 \
  --netG sr_mobile_composite \
