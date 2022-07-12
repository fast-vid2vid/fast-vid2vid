import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import time
from tqdm import tqdm
import torchvision
from util.keyframes import keyframe_selector_infer
from util.interploate_motion_ffmpeg import motion_compensation

if __name__ == '__main__':  
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    if opt.dataset_mode == 'temporal':
        opt.dataset_mode = 'test'

    # cal the keyframes probs
    keyframes = keyframe_selector_infer(opt.dataroot,opt.dataset_mode)



    data_loader = CreateDataLoader(opt,keyframes=keyframes)
    dataset = data_loader.load_data()
    model = create_model(opt,has_teacher=False)
    visualizer = Visualizer(opt)
    input_nc = 1 if opt.label_nc != 0 else opt.input_nc
    save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    print('Doing %d frames' % len(dataset))




    pbar = tqdm(dataset)
    for i, data in enumerate(pbar):
        #if i >= 100:#opt.how_many:
        #    break   
        if data['change_seq'] or i==0:
            model.fake_B_prev = None
            idx_prob = i//28
            #probs = keyframes_probs[idx_prob]
        seq_idx = i%28
        _, _, height, width = data['A'].size()
        A = Variable(data['A']).view(1, -1, input_nc, height, width)
        B = Variable(data['B']).view(1, -1, opt.output_nc, height, width) if len(data['B'].size()) > 2 else None
        inst = Variable(data['inst']).view(1, -1, 1, height, width) if len(data['inst'].size()) > 2 else None
        
        
        differ = B[0,-1]-B[0,-2]
        differ = differ.abs().mean(0).unsqueeze(0)


        
        generated = model.inference(A, B, inst)


        if opt.label_nc != 0:
            real_A = util.tensor2label(generated[1], opt.label_nc)
        else: 
            c = 3 if opt.input_nc == 3 else 1
            real_A = util.tensor2im(generated[1][:c], normalize=False)    


        img_path = data['A_path']



        if 'nosave' not in opt.results_dir:
            visual_list = [
                        ('fake_B', util.tensor2im(generated[0].data[0])),                   
                    
                        ]
                        
            visuals = OrderedDict(visual_list) 
            
            
            visualizer.save_images(save_dir+'/keyframes_bk/', visuals, img_path)

    #do the ffmepg motion interpolation...
    motion_compensation([save_dir+'/keyframes_bk'],save_dir)
