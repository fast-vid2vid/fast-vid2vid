
python inference.py --name test_vid2vid --label_nc 35 --loadSize 512 \
--use_instance --fg --use_real_img --netG sr_mobile_composite \
--dataroot datasets/city/ \
--load_pretrain checkpoints/city/ \
--results_dir results/city/
