
python inference.py --name test_vid2vid \
--dataroot datasets/face/ --dataset_mode face \
--input_nc 15 --loadSize 512 --use_real_img \
 --netG sr_mobile_composite \
--load_pretrain checkpoints/face \
--results_dir results/face/
