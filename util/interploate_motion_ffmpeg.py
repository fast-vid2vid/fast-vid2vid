import glob
import os
import numpy as np
import glob
import random

def command(dir,gap,save_dir):
    cmd1 = "ffmpeg -threads 16  -r 1 -pattern_type glob -i '" + dir+"/*.jpg'"+ " results/tmp_pr.mp4 -y" 
    cmd2 = 'ffmpeg -threads 16 -i results/tmp_pr.mp4 -filter_complex "minterpolate='+'fps='+str(gap)+':mi_mode=mci:mc_mode=obmc:me=epzs" results/interpolate_out_pr.mp4 -y'

    cmd2_2 = 'ffmpeg -threads 16 -r 1 -i results/interpolate_out_pr.mp4 results/interpolate_out_r_1_pr.mp4 -y'
    cmd3 = 'ffmpeg -threads 16 -i results/interpolate_out_r_1_pr.mp4 -r 1 '+save_dir+'/fake_B_%5d.jpg -y'

    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd2_2)
    os.system(cmd3)
    os.system('rm -f results/*.mp4')
    

paths = ['']# the gap results dir
motion_dir_root = '' #the target results dir
total_frames = 28
def motion_compensation(paths,motion_dir_root,total_frames=28):
    for path in paths:

        motion_dir = motion_dir_root +'/'+ path.split('/')[-1].replace('_bk','_motion_new')+'/'
        
        dirs = sorted(glob.glob(path+'/*'))
        for dir in dirs:
            file_len = len(glob.glob(dir+'/*'))
            if file_len>=total_frames:
                continue
            gap = np.ceil(total_frames/file_len)
            
            save_dir = motion_dir+dir.split('/')[-1]+'/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


            command(dir,gap,save_dir)
            del_files = sorted(glob.glob(save_dir+'/*'))[::-1]
            while len(del_files)<total_frames:
                gap+=1
                command(dir,gap,save_dir)
                del_files = sorted(glob.glob(save_dir+'/*'))[::-1]

            if len(del_files)>total_frames:
                del_num = len(del_files)-total_frames
                jump = int(total_frames/del_num)
                for i in range(del_num):
                    idx = random.choice(range(i*jump,(i+1)*jump))
                    if idx == 0:
                        idx +=1
                    if idx >=len(del_files):
                        idx -= 2
                    os.system('rm -f '+del_files[idx])



