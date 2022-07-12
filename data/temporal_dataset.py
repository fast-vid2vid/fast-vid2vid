
import os.path
import random
import torch
from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params
from data.image_folder import make_grouped_dataset, check_path_valid
from PIL import Image
import numpy as np

class TemporalDataset(BaseDataset):
    def initialize(self, opt,keyframes=None):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'A', opt.phase ) 
        self.dir_B = os.path.join(opt.dataroot, 'B',opt.phase)
        self.A_is_label = self.opt.label_nc != 0

        self.A_paths = sorted(make_grouped_dataset(self.dir_A))
        self.B_paths = sorted(make_grouped_dataset(self.dir_B))
        check_path_valid(self.A_paths, self.B_paths)
        if opt.use_instance:                
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
        print ('Data root_path:',self.dir_A,self.dir_B,self.dir_inst)
        
        self.n_of_seqs = len(self.A_paths)                 # number of sequences to train       
        self.seq_len_max = max([len(A) for A in self.A_paths])        
        self.n_frames_total = self.opt.n_frames_total      # current number of frames to train in a single iteration

    def __getitem__(self, index):
        tG = self.opt.n_frames_G
        A_paths = self.A_paths[index % self.n_of_seqs]
        B_paths = self.B_paths[index % self.n_of_seqs]                
        if self.opt.use_instance:
            I_paths = self.I_paths[index % self.n_of_seqs]                        
        
        # setting parameters
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(A_paths), index)     

        # setting transformers
        B_img = Image.open(B_paths[start_idx]).convert('RGB')        
        params = get_img_params(self.opt, B_img.size)          
        transform_scaleB = get_transform(self.opt, params)
        transform_scaleA = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) if self.A_is_label else transform_scaleB

        # read in images
        A = B = inst = 0
        for i in range(n_frames_total):            
            A_path = A_paths[start_idx + i * t_step]
            B_path = B_paths[start_idx + i * t_step]            
            Ai = self.get_image(A_path, transform_scaleA, is_label=self.A_is_label)            
            Bi = self.get_image(B_path, transform_scaleB)
            
            A = Ai if i == 0 else torch.cat([A, Ai], dim=0)            
            B = Bi if i == 0 else torch.cat([B, Bi], dim=0)            

            if self.opt.use_instance:
                I_path = I_paths[start_idx + i * t_step]                
                Ii = self.get_image(I_path, transform_scaleA) * 255.0
                inst = Ii if i == 0 else torch.cat([inst, Ii], dim=0)                

        return_list = {'A': A, 'B': B, 'inst': inst, 'A_path': A_path, 'B_paths': B_path, 'change_seq': self.change_seq}
        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        A_img = Image.open(A_path)        
        A_scaled = transform_scaleA(A_img)
        if is_label:
            A_scaled *= 255.0
        return A_scaled

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'TemporalDataset'