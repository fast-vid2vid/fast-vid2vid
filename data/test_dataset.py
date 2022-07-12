import os.path
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np

class TestDataset(BaseDataset):
    def initialize(self, opt,keyframes=None):

        self.opt = opt
        self.root = opt.dataroot
        #opt.phase = 'train'
        self.dir_A = os.path.join(opt.dataroot, 'A', opt.phase )
        self.dir_B = os.path.join(opt.dataroot, 'B',opt.phase)
        self.use_real = opt.use_real_img
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        if self.use_real:
            self.B_paths = sorted(make_grouped_dataset(self.dir_B))
            check_path_valid(self.A_paths, self.B_paths)
        if self.opt.use_instance:                
            self.dir_inst = os.path.join(opt.dataroot,'Inst', opt.phase )
            self.I_paths = sorted(make_grouped_dataset(self.dir_inst))
            check_path_valid(self.A_paths, self.I_paths)

        A_paths,B_paths,I_paths = [],[],[]
        if keyframes is not None:
            for i in range(len(self.A_paths)):
                #first ones
                A_paths.append(self.A_paths[i][:opt.n_frames_G])
                B_paths.append(self.B_paths[i][:opt.n_frames_G])
                I_paths.append(self.I_paths[i][:opt.n_frames_G])
                for j in keyframes[i]:
                    A_paths[i].append(self.A_paths[i][j])
                    B_paths[i].append(self.B_paths[i][j])
                    I_paths[i].append(self.I_paths[i][j])
                    
                #final one
                A_paths[i].append(self.A_paths[i][-1])
                B_paths[i].append(self.B_paths[i][-1])
                I_paths[i].append(self.I_paths[i][-1])
            self.A_paths,self.B_paths,self.I_paths = A_paths,B_paths,I_paths
        print ('video numbers:',len(self.A_paths),len(self.B_paths))

        self.init_frame_idx(self.A_paths)
        print ('Data root_path:',self.dir_A,self.dir_B,self.dir_inst)

    def __getitem__(self, index):
        self.A, self.B, self.I, seq_idx = self.update_frame_idx(self.A_paths, index)
        tG = self.opt.n_frames_G
              
        A_img = Image.open(self.A_paths[seq_idx][0]).convert('RGB')        
        params = get_img_params(self.opt, A_img.size)
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB
        frame_range = list(range(tG)) if self.A is None else [tG-1]
           
        for i in frame_range:                                                   
            A_path = self.A_paths[seq_idx][self.frame_idx + i]            
            Ai = self.get_image(A_path, transform_scaleA, is_label=self.A_is_label)            
            self.A = concat_frame(self.A, Ai, tG)

            if self.use_real:
                B_path = self.B_paths[seq_idx][self.frame_idx + i]
                Bi = self.get_image(B_path, transform_scaleB)                
                self.B = concat_frame(self.B, Bi, tG)
            else:
                self.B = 0

            if self.opt.use_instance:
                I_path = self.I_paths[seq_idx][self.frame_idx + i]
                Ii = self.get_image(I_path, transform_scaleA) * 255.0                
                self.I = concat_frame(self.I, Ii, tG)
            else:
                self.I = 0

        self.frame_idx += 1        
        return_list = {'A': self.A, 'B': self.B, 'inst': self.I, 'A_path': A_path, 'change_seq': self.change_seq}
        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):        
        return sum(self.frames_count)

    def n_of_seqs(self):        
        return len(self.A_paths)

    def name(self):
        return 'TestDataset'
