import os, h5py, sys
import cv2
import json
import pdb, tqdm
import numpy as np
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *

audio_data = h5py.File(AUDIO_FEATURES_H5PY_PATH)

if not os.path.exists(LABEL_DIRECTORY):
    os.makedirs(LABEL_DIRECTORY)

# Split data to train data, valid data and test data
def splitdata(path, train_num, val_num):
    lst = os.listdir(path)
    name = []
    for ele in lst:
        name.append(os.path.splitext(ele)[0])

    print(len(name))
    print(name[0:100])
    name = np.random.permutation(name)
    print(name[0:100])

    train = name[0:train_num]
    val = name[train_num:train_num+val_num]
    test = name[train_num+val_num:]
    np.savez('msvd_dataset',train=train, val=val, test=test)


def get_total_frame_number(fn):
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print("could not open :",fn)
        sys.exit() 
    length = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def get_total_audio_frame_number(fn):
    aud = np.load(fn)
    return aud.shape[0]


def getlist(Dir):
    pylist = []
    for root, dirs, files in os.walk(Dir):
        for ele in files:
            pylist.append(root+ele)
            
#     for root, dirs, files in os.walk(os.path.join(Dir,'test')):
#         for ele in files:
#             pylist.append(root+'/'+ele)
            
            
#     for root, dirs, files in os.walk(os.path.join(Dir,'validation')):
#         for ele in files:
#             pylist.append(root+'/'+ele)
    # print pylist
    return pylist

def get_frame_list(frame_num,NUMBER_OF_SEGMENTS):

    num_clip8 = np.floor(frame_num/8)-1
    clip_per_seg = float(num_clip8/NUMBER_OF_SEGMENTS)

    seg_left = np.zeros(NUMBER_OF_SEGMENTS)
    seg_right = np.zeros(NUMBER_OF_SEGMENTS)

    for i in range(NUMBER_OF_SEGMENTS):
        seg_left[i] = int(i * clip_per_seg)
        seg_right[i] = int((i+1) * clip_per_seg)
        if seg_right[i] > num_clip8:
            seg_right[i] = num_clip8
            
    clip_num_list = []
    for i in range(NUMBER_OF_SEGMENTS):
        clip_num_list.append([ seg_left[i],seg_right[i] ])

    return clip_num_list


def get_audio_frame_list(frame_num,NUMBER_OF_SEGMENTS):
    
    clip_per_seg = float((1.0*frame_num)/NUMBER_OF_SEGMENTS)

    seg_left = np.zeros(NUMBER_OF_SEGMENTS)
    seg_right = np.zeros(NUMBER_OF_SEGMENTS)

    for i in range(NUMBER_OF_SEGMENTS):
        seg_left[i] = int(i * clip_per_seg)
        seg_right[i] = int((i+1) * clip_per_seg)
        if seg_right[i] >= frame_num:
            seg_right[i] = frame_num-1
            
    clip_num_list = []
    for i in range(NUMBER_OF_SEGMENTS):
        clip_num_list.append([ seg_left[i],seg_right[i] ])

    return clip_num_list

def get_label_list(fname):
    try:
        frame_len = get_total_frame_number(fname)
        alist = list(fname.split('/')[-1].split('.')[0])
        # alist[0] = 'a'
        
        aname = fname.split('/')[-1].split('.')[0]
       
        aud_len = audio_data[aname].shape[0]    

        aud_frame_list = get_audio_frame_list(aud_len,NUMBER_OF_SEGMENTS)
       
        frame_list = get_frame_list(frame_len,NUMBER_OF_SEGMENTS)
        
        if frame_list == -1:
            return
        label_list = [-1]*len(frame_list)
        label_list[-1] = 0
        fname = fname.split('/')[-1]
        outfile = LABEL_DIRECTORY+str(aname)+'.json'
        if not os.path.isfile(outfile):
            json.dump([frame_list, aud_frame_list, label_list], open(outfile,"w"))
    except:
        print("error")
        print(fname)
        return

if __name__=='__main__':
    b = getlist(PROCESSED_VIDEOS_DIRECTORY)
    count = 0
    for ele in b:
        fname = ele
        if not os.path.isfile(LABEL_DIRECTORY+str(fname)+'.json'):
            print(fname)
            get_label_list(fname)
        count += 1
    print(len(b))
    print(count)
