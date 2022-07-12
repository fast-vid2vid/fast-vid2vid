
import pdb
from re import T
import time
import os
import torch
from subprocess import call

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model, create_optimizer, init_params, save_models, update_models
import util.util as util
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
from tqdm import tqdm



def train():

    opt = TrainOptions().parse()



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
    models = create_model(opt,False)
    
    #[modelG, modelD, flowNet] = models
    modelG,_, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer(opt, models)
    


    ### set parameters    
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, \
        start_epoch, epoch_iter, print_freq, total_steps, iter_path = init_params(opt, modelG, modelD, data_loader)
    visualizer = Visualizer(opt)    
    best_loss = 9999999
    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()    
        print (opt.name,'\n Epoch:',epoch)
        pbar = tqdm(dataset)
        epoch_loss = 0
        
        for idx, data in enumerate(pbar):
            
            sum_loss = 0        
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            log = False
            if idx%30 == 0:
                log = True
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # whether to collect output images
            save_fake = total_steps % opt.display_freq == 0
            n_frames_total, n_frames_load, t_len = data_loader.dataset.init_data_params(data, n_gpus, tG)
            fake_B_prev_last, frames_all = data_loader.dataset.init_data(t_scales)
            for i in range(0, n_frames_total, n_frames_load):

                input_A, input_B, inst_A = data_loader.dataset.prepare_data(data, i, input_nc, output_nc)
        
                ###################################### Forward Pass ##########################
                ####### generator      
                if 'pyramid' in opt.name:
                    fake_B_, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last,fake_B_list,fake_B_raw_list = modelG(input_A, input_B, inst_A, fake_B_prev_last)
                    fake_B = fake_B_list
                    fake_B_raw = fake_B_raw_list
                else:
                    fake_B, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last = modelG(input_A, input_B, inst_A, fake_B_prev_last)
                #_, _, _, _, _, _, _ = modelG_teacher(input_A, input_B, inst_A, fake_B_prev_last)
                #print (fake_B.size())
                #assert 1==2
                ####### discriminator            
                ### individual frame discriminator          
                real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]   # the collection of previous and current real frames


                flow_ref, conf_ref = flowNet(real_B, real_B_prev)       # reference flows and confidences     
                
                fake_B_prev = modelG.module.compute_fake_B_prev(real_B_prev, fake_B_prev_last, fake_B)
                fake_B_prev_last = fake_B_last

                losses =  modelD(0, reshape([real_B, fake_B[-1], fake_B_raw[-1], real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref]))
                for i_fb in range(len(fake_B)-1):
                    losses_tmp = modelD(0, reshape([real_B, fake_B[i_fb], fake_B_raw[i_fb], real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref]))
                    for i_loss in range(len(losses_tmp)):
                        if losses_tmp[i_loss] is not None:
                            losses[i_loss] += losses_tmp[i_loss]
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))


                frames_all, frames_skipped = modelD.module.get_all_skipped_frames(frames_all, \
                        real_B, fake_B[-1], flow_ref, conf_ref, t_scales, tD, n_frames_load, i, flowNet)      

                # run discriminator for each temporal scale
                loss_dict_T = []
                for s in range(t_scales):                
                    if frames_skipped[0][s] is not None:                        
                        losses = modelD(s+1, [frame_skipped[s] for frame_skipped in frames_skipped])
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

                # collect losses
                loss_G, loss_D, loss_D_T, t_scales_act = modelD.module.get_losses(loss_dict, loss_dict_T, t_scales)
                sum_loss+=loss_G


                ###################################### Backward Pass #################################                 
                # update generator weights     
                loss_backward(opt, loss_G, optimizer_G)                

                # update individual discriminator weights                
                loss_backward(opt, loss_D, optimizer_D)

                # update temporal discriminator weights
                for s in range(t_scales_act):                    
                    loss_backward(opt, loss_D_T[s], optimizer_D_T[s])

                if i == 0: 
                    fake_B_first = fake_B[0, 0]   # the first generated image in this sequence
                visuals = util.save_all_tensors(opt, real_A, fake_B[-1], fake_B_first, fake_B_raw[-1], real_B, flow_ref, conf_ref, flow, weight, modelD)  
                
                

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
            if epoch_iter > dataset_size - opt.batchSize:
                epoch_iter = 0
                break
            sum_loss /= t_len
            epoch_loss+=sum_loss
        # end of epoch 

        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        if epoch_loss<best_loss:
            best_loss = epoch_loss
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