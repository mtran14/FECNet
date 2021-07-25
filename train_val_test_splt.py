import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shutil import copyfile
import os

data_type = 'cremad'
if(data_type == 'mosi'):
    meta_data_path = '/shares/perception-working/minh/Sentiment_Annotation_2021.csv'
    input_video_path = '/shares/perception-working/minh/CMU_MOSI/'
    output_dir = '/shares/perception-working/minh/fecnet/mosi/'
    data = pd.read_csv(meta_data_path).values
    user_ids = []
    for row in data:
        user_ids.append(row[0][:-5])
    X = list(set(user_ids))
    y = np.ones(len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    print(len(X_train), len(X_test), len(X_val))

    output_path_train = output_dir + 'train/'
    output_path_test = output_dir + 'test/'
    output_path_val = output_dir + 'val/'

    for vfile in os.listdir(input_video_path):
        current_video_path = os.path.join(input_video_path, vfile)
        current_output_path = None
        for pid in X_train:
            if(pid in vfile):
                current_output_path = output_path_train
        for pid in X_test:
            if(pid in vfile):
                current_output_path = output_path_test
        for pid in X_val:
            if(pid in vfile):
                current_output_path = output_path_val
        if(current_output_path):
            current_output_video_path = os.path.join(current_output_path, vfile)
            copyfile(current_video_path, current_output_video_path)
        else:
            print('cannot match: ', vfile)
elif(data_type == 'cremad'):
    meta_data_path = '/shares/perception-working/minh/CREMA-D/tabulatedVotes.csv'
    input_video_path = '/shares/perception-working/minh/CREMA-D/VideoFlash/'
    output_dir = '/shares/perception-working/minh/fecnet/cremad/'

    output_path_train = output_dir + 'train/'
    output_path_test = output_dir + 'test/'
    output_path_val = output_dir + 'val/'

    X = user_ids = list(range(1001, 1092))
    X = [str(x) for x in X]
    y = np.ones(len(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    print(len(X_train), len(X_test), len(X_val))
    for vfile in os.listdir(input_video_path):
        current_video_path = os.path.join(input_video_path, vfile)
        current_output_path = None
        for pid in X_train:
            if(pid in vfile):
                current_output_path = output_path_train
        for pid in X_test:
            if(pid in vfile):
                current_output_path = output_path_test
        for pid in X_val:
            if(pid in vfile):
                current_output_path = output_path_val
        if(current_output_path):
            current_output_video_path = os.path.join(current_output_path, vfile)
            copyfile(current_video_path, current_output_video_path)
        else:
            print('cannot match: ', vfile)
