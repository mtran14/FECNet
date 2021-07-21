#!/home/ICT2000/mtran/miniconda3/envs/torch/bin/python
import os, sys
import numpy as np
import torch
from models.FECNet import FECNet
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from moviepy.editor import *
import pandas as pd
from multiprocessing import Process, Manager, Pool
import multiprocessing
import random


output_path = "/data/perception-temp/voxceleb2/fecnet/train/"


def extractFecNet(files):
    random.shuffle(files)
    model = FECNet('FECNet.pt')
    mtcnn = MTCNN(image_size=224)
    for file in files:
        try:
            file_path_split = file.split("/")
            id1, id2, fname = file_path_split[-3], file_path_split[-2], file_path_split[-1]
            output_file_name = id1 + '_' + id2 + '_' + fname.split('.')[0] + '.csv'
            output_file_path = os.path.join(output_path, output_file_name)
            
            if(os.path.isfile(output_file_path)):
                continue
            
            vidcap = VideoFileClip(file)
            frames = list(vidcap.iter_frames(fps=5))
            
            faces, prob = mtcnn(frames, return_prob=True)
            faces = [t.numpy() for t in faces]
            faces = np.array(faces)
            if faces.any():
                faces = torch.Tensor(faces).view(-1,3,224,224)
                emb = model(faces)
                emb = emb.detach().numpy()
                pd.DataFrame(emb).to_csv(output_file_path, header=None, index=False)
        except:
            continue
            
def extractFecNetSingle(file):
    # try:
    model = FECNet('FECNet.pt')
    mtcnn = MTCNN(image_size=224)
    try:
        file_path_split = file.split("/")
        id1, id2, fname = file_path_split[-3], file_path_split[-2], file_path_split[-1]
        output_file_name = id1 + '_' + id2 + '_' + fname.split('.')[0] + '.csv'
        output_file_path = os.path.join(output_path, output_file_name)

        if(os.path.isfile(output_file_path)):
            return

        vidcap = VideoFileClip(file)
        frames = list(vidcap.iter_frames(fps=5))

        faces, prob = mtcnn(frames, return_prob=True)
        faces = [t.numpy() for t in faces]
        faces = np.array(faces)
        print(faces.shape)
        if faces.any():
            faces = torch.Tensor(faces).view(-1,3,224,224)
            emb = model(faces)
            emb = emb.detach().numpy()
            pd.DataFrame(emb).to_csv(output_file_path, header=None, index=False)
            print(output_file_path)
    except:
        return

            
def fecnet_extract_in_parallel(concurreny_count, files, fn):
    Processes = []
    # files_  =  [files[(i* (len(files)//concurreny_count)):((i+1)* (len(files)//concurreny_count))]    for i in range(concurreny_count)]
    # leftovers  =  files[(concurreny_count * (len(files)//concurreny_count))  :  len(files)]
    # for i in range(len(leftovers)):    files_[i] += [leftovers[i]]
    files_ = np.array_split(files, len(files)//concurreny_count)
    random.shuffle(files_)
    for  files_list_  in files_:
        p = Process(target=fn, args=(files_list_, files_list_))
        Processes.append(p)
        p.start()
    # block until all the threads finish (i.e. block until all function_x calls finish)
    for t in Processes:    t.join()
        
target_chunk_size = 5
concurreny_count = 10
meta_file_path = "../mm_ted/data/file_paths_fm.csv"
files = pd.read_csv(meta_file_path, header=None).values[:,0]

# extractFecNet(files)
# fecnet_extract_in_parallel(concurreny_count, files, extractFecNet)
# concurreny_count = len(files) // target_chunk_size
# 
# files_ = np.array_split(files, concurreny_count)
# random.shuffle(files_)
# for files in files_:
#     extractFecNetMultiVid(files)
random.shuffle(files)
pool = multiprocessing.Pool(40)
pool.map(extractFecNetSingle, files)
pool.close()
