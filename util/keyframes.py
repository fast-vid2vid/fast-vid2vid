
from glob import glob
import cv2
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from tqdm import tqdm
USE_THRESH = False
THRESH = 0.6
USE_TOP_ORDER = False
USE_LOCAL_MAXIMA = True
NUM_TOP_FRAMES = 20


videopath = ''#sys.argv[1]
#Directory to store the processed frames
dir = None
name_npy = 'keyframes_window_3.npy'
len_window = 5#int(name_npy.split('window_')[-1][0])
print ('len-window',len_window)
vid_keyframes_idx = []
sum_length = 0
video_dirs = sorted(glob('../vid2vid/datasets/city/A/*'))


def smooth(x, window_len=3, window='hanning'):

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        return x

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise( ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]


    if window == 'flat':  
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]




class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(a, b):
   x = (b - a) / max(a, b)
   print(x)
   return x

def read_frames( videopath='./',is_images_dir = True):
    if not is_images_dir:
        cap = cv2.VideoCapture(str(videopath))


        curr_frame = None
        prev_frame = None

        frame_diffs = []
        frames = []
        ret, frame = cap.read()
        i = 1

        while(ret):
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            curr_frame = luv
            if curr_frame is not None and prev_frame is not None:
                #logic here
                diff = cv2.absdiff(curr_frame, prev_frame)
                count = np.sum(diff)
                frame_diffs.append(count)
                frame = Frame(i, frame, count)
                frames.append(frame)
            prev_frame = curr_frame
            i = i + 1
            ret, frame = cap.read()

        cap.release()
        
    else:
        files = sorted(glob(videopath+'/*.png'))
        frame_diffs = []
        frames = []
        i = 1
        curr_frame = None
        prev_frame = None

        for f in files:
            frame = cv2.imread(f)
            luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
            if 'B' in f:
                luv = cv2.resize(luv,(512,512))
            curr_frame = luv
            if curr_frame is not None and prev_frame is not None:
                #logic here
                diff = cv2.absdiff(curr_frame, prev_frame)
                count = np.sum(diff)
                frame_diffs.append(count)
                frame = Frame(i, frame, count)
                frames.append(frame)
            prev_frame = curr_frame
    return frames,frame_diffs

def extract_single_vid_keyframes(videopath,len_windows=3,is_images_dir=True):
    frames,frame_diffs = read_frames(videopath,is_images_dir)
    if USE_TOP_ORDER:
        # sort the list in descending order
        frames.sort(key=operator.attrgetter("value"), reverse=True)
        for keyframe in frames[:NUM_TOP_FRAMES]:
            name = "frame_" + str(keyframe.id) + ".jpg"
            cv2.imwrite(dir + "/" + name, keyframe.frame)

    if USE_THRESH:
        print("Using Threshold")
        for i in range(1, len(frames)):
            if (rel_change(np.float(frames[i - 1].value), np.float(frames[i].value)) >= THRESH):
                name = "frame_" + str(frames[i].id) + ".jpg"
                cv2.imwrite(dir + "/" + name, frames[i].frame)


    if USE_LOCAL_MAXIMA:
        diff_array = np.array(frame_diffs)
        sm_diff_array = smooth(diff_array, len_windows)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
        frame_indexes = [f+1 for f in frame_indexes]

    return frame_indexes


def keyframe_selector_infer(video_dirs,mode,len_windows=3):
    
    sum_length = 0
    if mode == 'face':
        video_dirs = sorted(glob(video_dirs+'test_semantics/*'))
    if mode == 'test':
        video_dirs =sorted( glob(video_dirs+'A/test/*'))
    vid_keyframes_idx = []
    for videopath in video_dirs:
        frame_indexes = extract_single_vid_keyframes(videopath,len_windows)
        sum_length += len(frame_indexes)
        vid_keyframes_idx.append(frame_indexes)

    vid_keyframes_idx = np.array(vid_keyframes_idx)
    return vid_keyframes_idx

def keyframe_selector(video_dirs,name_npy=name_npy):
    vid_keyframes_idx = []
    for videopath in tqdm(video_dirs):
        frame_indexes = extract_single_vid_keyframes(videopath)
        sum_length += len(frame_indexes)
        vid_keyframes_idx.append(frame_indexes)

    vid_keyframes_idx = np.array(vid_keyframes_idx)

    np.save(name_npy, vid_keyframes_idx)
    print ('========================',name_npy,vid_keyframes_idx.shape,sum_length,'===============================')
        

