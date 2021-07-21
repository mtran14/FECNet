import os, sys
import numpy as np
import torch
from models.FECNet import FECNet
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from moviepy.editor import *
import pandas as pd
from multiprocessing import Process, Manager
import random

output_path = "/shares/perception-temp/voxceleb2/fecnet/train/"

def extractFecNetMultiVid(files):
    combine_faces = []
    chunk_sizes = []
    idx = 0
    output_file_names = []
    random.shuffle(files)
    shapes = []
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
            faces = np.array(faces) # n_frames*3*224*224
            
            combine_faces.append(faces)
            idx = idx + faces.shape[0]
            chunk_sizes.append(idx)
            output_file_names.append(output_file_path)
            shapes.append(faces.shape[0])
        except:
            continue
    all_faces = np.concatenate(combine_faces, axis=0)
    all_faces = torch.Tensor(all_faces).view(-1,3,224,224)
    emb = model(all_faces.cuda()).cpu()
    emb = emb.detach().numpy()
    emb_split = np.split(emb, chunk_sizes)
    for i in range(len(output_file_names)):
        dout, fout = emb_split[i], output_file_names[i]
        print(dout.shape[0], shapes[i])
        pd.DataFrame(dout).to_csv(fout, header=None, index=False)
    

def extractFecNet(files, buff):
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
                emb = model(faces.cuda()).cpu()
                emb = emb.detach().numpy()
                pd.DataFrame(emb).to_csv(output_file_path, header=None, index=False)
        except:
            continue
            
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
    
target_chunk_size = 50
concurreny_count = 10
meta_file_path = sys.argv[1]
files = pd.read_csv(meta_file_path, header=None).values[:,0]

fecnet_extract_in_parallel(concurreny_count, files, extractFecNet)
# concurreny_count = len(files) // target_chunk_size
# 
# files_ = np.array_split(files, concurreny_count)
# random.shuffle(files_)
# for files in files_:
#     extractFecNetMultiVid(files)
