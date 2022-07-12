import os
import glob
from skimage import io
import numpy as np
import dlib
import sys
from tqdm import tqdm
import concurrent.futures as cf

phase = sys.argv[1]
dataset_path = ''
faces_folder_path = os.path.join(dataset_path, phase + '_img/')
predictor_path = ''
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def func(img_name):
    img = io.imread(img_name)
    dets = detector(img, 1)
    if len(dets) > 0:
        shape = predictor(img, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b,0] = shape.part(b).x
            points[b,1] = shape.part(b).y

        save_name = os.path.join(save_path, os.path.basename(img_name)[:-4] + '.txt')
        np.savetxt(save_name, points, fmt='%d', delimiter=',')

img_paths = sorted(glob.glob(faces_folder_path + '*'))

for i in tqdm(range(len(img_paths))):
    f = img_paths[i]
    #print("Processing video: {}".format(f))
    save_path = os.path.join(dataset_path, phase + '_keypoints', os.path.basename(f))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for img_name in tqdm(sorted(glob.glob(os.path.join(f, '*.png')))):
        img = io.imread(img_name)
        dets = detector(img, 1)
        if len(dets) > 0:
            shape = predictor(img, dets[0])
            points = np.empty([68, 2], dtype=int)
            for b in range(68):
                points[b,0] = shape.part(b).x
                points[b,1] = shape.part(b).y

            save_name = os.path.join(save_path, os.path.basename(img_name)[:-4] + '.txt')
            np.savetxt(save_name, points, fmt='%d', delimiter=',')


