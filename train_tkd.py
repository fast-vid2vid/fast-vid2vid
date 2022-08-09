from re import T
import time
import os
import torch
from subprocess import call

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model, create_optimizer, init_params, save_models, update_models
import util.util as util
#from util.common import shrink
from util.visualizer import Visualizer
from tqdm import tqdm
from models import networks
from util.util import tensor2flow
import random
import pdb
import numpy as np
import copy
import torch.nn as nn

def jump(data2,opt):
    data = copy.deepcopy(data2)
    _, n_frames_total, height, width = data['inst'].size()
    input_nc,output_nc = opt.input_nc,opt.output_nc


    gap = random.choice(range(2,n_frames_total-2))

    t_len = int(np.ceil(n_frames_total/gap))
    while t_len<opt.n_frames_G:
        gap = random.choice(range(2,n_frames_total-2))
        t_len = int(np.ceil(n_frames_total/gap))      
          

    data['A'] = data['A'].view(-1, n_frames_total, input_nc, height, width)[:,::gap]
    data['B'] = data['B'].view(-1, n_frames_total, output_nc, height,width)[:,::gap]    
    data['inst'] = data['inst'].view(-1, n_frames_total, 1, height, width)[:,::gap]

    data['A'] = data['A'].reshape(-1, t_len * input_nc, height, width)
    data['B'] = data['B'].reshape(-1, t_len * output_nc, height,width)    
    data['inst'] = data['inst'].reshape(-1, t_len* 1, height, width)

    return data,gap

class KDLOSS(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = networks.VGGLoss(opt.gpu_ids[0])
        self.opt = opt

    def forward(self,fake_B_student,fake_B_teacher):

        kd_vgg_loss = self.criterionVGG(fake_B_student[0], fake_B_teacher[0])
        kd_l1_loss = self.criterionFeat(fake_B_student, fake_B_teacher)

        return [kd_l1_loss,kd_vgg_loss]
def train():
    opt = TrainOptions().parse()


    criterionFeat = torch.nn.L1Loss()
    criterionVGG = networks.VGGLoss(opt.gpu_ids[0])
    criterionI3D = networks.I3DLoss().cuda()
    criterionFlow = networks.MaskedL1Loss()
    
    KD_loss = KDLOSS(opt).cuda()
    KD_loss = torch.nn.DataParallel(KD_loss)
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1

    ### initialize dataset
    data_loader = CreateDataLoader(opt)

    dataset = data_loader.load_data()
    dataset_size = len(data_loader)    
    print('#training videos = %d' % dataset_size)


    ### initialize models
    models = create_model(opt)
    
    [modelG,modelG_teacher, modelD, flowNet] = models
    modelG,modelG_teacher, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer(opt, models)
    for p in modelG_teacher.parameters():
        p.requires_grad = False


    ### set parameters    
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, \
        start_epoch, epoch_iter, print_freq, total_steps, iter_path = init_params(opt, modelG, modelD, data_loader)
    visualizer = Visualizer(opt)    
    best_loss = 9999999
    cur_val_num = 0
    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()    
        print ('Epoch:',epoch)
        pbar = tqdm(dataset)
        epoch_loss_G = epoch_loss_G_len = 0


        for idx, data in enumerate(pbar):        
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            log = False
            if idx%30 == 0:
                log = True
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
            n_frames_total_teacher, n_frames_load_teacher, t_len_teacher = data_loader.dataset.init_data_params(data, n_gpus, tG)
            fake_B_prev_last_teacher, frames_all_teacher = data_loader.dataset.init_data(t_scales)

            data_jump,gap = jump(data,opt)
            

            with torch.no_grad():
                for i in range(0, n_frames_total_teacher, n_frames_load_teacher):
                    input_A, input_B, inst_A = data_loader.dataset.prepare_data(data, i, input_nc, output_nc)
                    

                    fake_B_teacher, fake_B_raw_teacher, flow_teacher, weight_teacher, real_A_teacher, real_Bp_teacher, fake_B_last_teacher = modelG_teacher(input_A, input_B, inst_A, fake_B_prev_last_teacher)
                    
                    real_B_prev_teacher, real_B_teacher = real_Bp_teacher[:, :-1], real_Bp_teacher[:, 1:]              
                    fake_B_prev_teacher = modelG.module.compute_fake_B_prev(real_B_prev_teacher, fake_B_prev_last_teacher, fake_B_teacher)
                    fake_B_prev_last_teacher = fake_B_last_teacher
                    if i == 0:
                        fake_B_teacher_dir = fake_B_teacher
                        flow_teacher_dir = flow_teacher
                        fake_B_prev_teacher_dir = fake_B_prev_teacher
                    else:
                        fake_B_teacher_dir = torch.cat((fake_B_teacher_dir,fake_B_teacher),axis=1)
                        flow_teacher_dir = torch.cat((flow_teacher_dir,flow_teacher),axis=1)
                        fake_B_prev_teacher_dir = torch.cat((fake_B_prev_teacher_dir,fake_B_prev_teacher),axis=1)
                        

            student_idx = list(range(data['inst'].shape[1])[::gap])
            student_idx = student_idx[opt.n_frames_G-1:]

            fake_B_teacher_dirs = []
            flow_teacher_dirs = []
            fake_B_prev_teacher_dirs = []
            for _ in range(len(student_idx)):
                sidx = student_idx[_]-opt.n_frames_G+1
                dir_idx = _//opt.max_frames_per_gpu
                if len(fake_B_teacher_dirs)< dir_idx+1:
                    fake_B_teacher_dirs.append(fake_B_teacher_dir[:,sidx:sidx+1])
                    flow_teacher_dirs.append(flow_teacher_dir[:,sidx:sidx+1])
                    fake_B_prev_teacher_dirs.append(fake_B_prev_teacher_dir[:,sidx:sidx+1])
                    
                else:
                    fake_B_teacher_dirs[dir_idx] = torch.cat((fake_B_teacher_dirs[dir_idx],fake_B_teacher_dir[:,sidx:sidx+1]),axis=1)
                    flow_teacher_dirs[dir_idx] = torch.cat((flow_teacher_dirs[dir_idx],flow_teacher_dir[:,sidx:sidx+1]),axis=1)
                    fake_B_prev_teacher_dirs[dir_idx] = torch.cat((fake_B_prev_teacher_dirs[dir_idx],fake_B_prev_teacher_dir[:,sidx:sidx+1]),axis=1)

            # whether to collect output images
            save_fake = total_steps % 30 == 0
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params(data_jump, n_gpus, tG)
            fake_B_prev_last, frames_all = data_loader.dataset.init_data(t_scales)      
            dir_idx = 0 
            for i in range(0, n_frames_total, n_frames_load):

                fake_B_teacher = fake_B_teacher_dirs[dir_idx]
                
                flow_teacher = flow_teacher_dirs[dir_idx]
                fake_B_prev_teacher = fake_B_prev_teacher_dirs[dir_idx]
                dir_idx += 1

                input_A, input_B, inst_A = data_loader.dataset.prepare_data(data_jump, i, input_nc, output_nc)


                fake_B, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last = modelG(input_A, input_B, inst_A, fake_B_prev_last,new_n_frames_load=input_A.shape[1]-2)

                

                ####### discriminator            
                ### individual frame discriminator          
                real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]   # the collection of previous and current real frames
                flow_ref, conf_ref = flowNet(real_B, real_B_prev)       # reference flows and confidences                
                fake_B_prev = modelG.module.compute_fake_B_prev(real_B_prev, fake_B_prev_last, fake_B)
                fake_B_prev_last = fake_B_last


                
                losses = modelD(0, reshape([real_B, fake_B, fake_B_raw, real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref]))
                

                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))

                
                ### temporal discriminator 
                frames_all, frames_skipped = modelD.module.get_all_skipped_frames(frames_all, \
                        real_B, fake_B, flow_ref, conf_ref, t_scales, tD, n_frames_load, i, flowNet)                                

                # run discriminator for each temporal scale
                loss_dict_T = []
                for s in range(t_scales):                
                    if frames_skipped[0][s] is not None:                        
                        losses = modelD(s+1, [frame_skipped[s] for frame_skipped in frames_skipped])
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))


                loss_G, loss_D, loss_D_T, t_scales_act = modelD.module.get_losses(loss_dict, loss_dict_T, t_scales)

                #kd loss

                if opt.use_temporal_loss:
                    fake_seq = torch.cat((fake_B_last[0],fake_B),dim=1)
            
                    #reshape_tensor = reshape([fake_B_prev_teacher, flow])
                    #fake_B_teacher_warp = modelD.module.resample(reshape_tensor[0],reshape_tensor[1]) 
                    
                    #to change
                    fake_teacher_seq = torch.cat((fake_B_last_teacher[0],fake_B_teacher),dim=1)

                    kd_i3d_loss = criterionI3D(fake_seq,fake_teacher_seq)
                    kd_i3d_loss *= 15

                    loss_F_Flow = criterionFeat(flow, flow_teacher) * opt.lambda_F     



                kd_loss_list = KD_loss(fake_B,fake_B_teacher)*2
            

                
                if opt.use_temporal_loss:
                    kd_loss_list += [kd_i3d_loss,loss_F_Flow]
                kd_loss_list = [x.mean() for x in kd_loss_list]*2
                

                loss_kd = sum(kd_loss_list)


                ###################################### Backward Pass #################################                 
                # update generator weights     
                loss_backward(opt, loss_G+loss_kd, optimizer_G)                  
                epoch_loss_G +=(loss_G+loss_kd)
                epoch_loss_G_len += 1
                # update individual discriminator weights                
                loss_backward(opt, loss_D, optimizer_D)

                # update temporal discriminator weights
                for s in range(t_scales_act):                    
                    loss_backward(opt, loss_D_T[s], optimizer_D_T[s])

                if i == 0: fake_B_first = fake_B[0, 0]   # the first generated image in this sequence

                visuals = util.save_all_tensors(opt, real_A, fake_B, fake_B_first, fake_B_raw, real_B, flow_ref, conf_ref, flow, weight, modelD)  


            
            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})            
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

                
            ### display output images
            if save_fake:                
                visuals = util.save_all_tensors(opt, real_A, fake_B, fake_B_first, fake_B_raw, real_B, flow_ref, conf_ref, flow, weight, modelD)                
                visualizer.display_current_results(visuals, epoch, total_steps)
                
            ### save latest model
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD)            

        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        epoch_loss_G /= epoch_loss_G_len
        if epoch_loss_G<best_loss:
            best_loss = epoch_loss_G
            print (opt.name,'\t Epoch:',epoch,'updating a best model with loss:',best_loss)
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=True,save_best=True)

        save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, end_of_epoch=True)
        update_models(opt, epoch, modelG, modelD, data_loader) 

def loss_backward(opt, loss, optimizer):
    optimizer.zero_grad()                
    if opt.fp16:
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss: 
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

def reshape(tensors):
    if tensors is None: return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]    
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

if __name__ == "__main__":
   train()
