#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import json
import pdb
import unicodedata
import glob
import os
import json
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *



path = PROCESSED_OUTPUT_DIRECTORY
 
ANNOTATION_TEST_PARSED_PATH = ANNOTATION_TEST_PARSED_PATH
MOVIE_LENGTH_FILE_PATH = MOVIE_LENGTH_FILE_PATH

if USE_FULL_SPLIT:
    splitdataset_path = FULL_SPLIT_DATASET_PATH
else:
    splitdataset_path = SMALL_SPLIT_DATASET_PATH

BATCH_SIZE = 20 # each mini batch will have BATCH_SIZE sentences
n_length = 128

if AUDIO_FEATURES_TO_USE == "vgg":
    video_fts_dim = CONT_CAPTION_FILE_FEATURE_DIMENSION_VGG
else:
    video_fts_dim = CONT_CAPTION_FILE_FEATURE_DIMENSION_MFCC



# In[2]:


movie_length_dict={}
with open(MOVIE_LENGTH_FILE_PATH)  as f:
    for l in f:
        content=l.rstrip().split(" ")
        content[1]=float(content[1])
        # content[2]=int(content[2])
        movie_length_dict[content[0]] = content[1]


# In[12]:


def create_merge_j_dict():
    train_j={}
    dataset = 'train'
    List = np.load(splitdataset_path)[dataset]#[0:1300]
    with open(ANNOTATION_TRAIN_PARSED_PATH) as f:
        for line in f:
            content=line.rstrip().split(" ")
            if content[0] in List:
                if content[0] not in train_j:
                    train_j[content[0]]={}
                    train_j[content[0]]['duration']=movie_length_dict[content[0]]
                    train_j[content[0]]['timestamps']=[]
                    s_time=float(content[1])
                    e_time=float(content[2].split("##")[0])
                    time_arr=[]
                    time_arr.append(s_time)
                    time_arr.append(e_time)
                    train_j[content[0]]['timestamps'].append(time_arr)
                    train_j[content[0]]['sentences']=[]
                    train_j[content[0]]['sentences'].append(line.rstrip().split("##")[1])
                else:
                    s_time=float(content[1])
                    e_time=float(content[2].split("##")[0])
                    time_arr=[]
                    time_arr.append(s_time)
                    time_arr.append(e_time)
                    train_j[content[0]]['timestamps'].append(time_arr)
                    train_j[content[0]]['sentences'].append(line.rstrip().split("##")[1])
    val_j={}
    dataset = 'val'
    List = np.load(splitdataset_path)[dataset]#[0:800]
    with open(ANNOTATION_TEST_PARSED_PATH) as f:
        for line in f:
            content=line.rstrip().split(" ")
            if content[0] in List:
                if content[0] not in val_j:
                    val_j[content[0]]={}
                    val_j[content[0]]['duration']=movie_length_dict[content[0]]
                    val_j[content[0]]['timestamps']=[]
                    s_time=float(content[1])
                    e_time=float(content[2].split("##")[0])
                    time_arr=[]
                    time_arr.append(s_time)
                    time_arr.append(e_time)
                    val_j[content[0]]['timestamps'].append(time_arr)
                    val_j[content[0]]['sentences']=[]
                    val_j[content[0]]['sentences'].append(line.rstrip().split("##")[1])
                else:
                    s_time=float(content[1])
                    e_time=float(content[2].split("##")[0])
                    time_arr=[]
                    time_arr.append(s_time)
                    time_arr.append(e_time)
                    val_j[content[0]]['timestamps'].append(time_arr)
                    val_j[content[0]]['sentences'].append(line.rstrip().split("##")[1])
    z = train_j.copy()
    z.update(val_j)
    return z
    


# In[13]:


merge_j=create_merge_j_dict();
#print(merge_j.keys())
# merge

# In[ ]:





# In[15]:


def getSegWeight(video_sec,ground_seg,label):

    label = np.array(label)
    segnum = np.where(label != -1)[0][0] + 1

    left = max(0,int((ground_seg[0] * 1.0 / video_sec) * segnum))
    right = min(len(label),int((ground_seg[1] * 1.0 / video_sec) * segnum))

    weight = np.zeros(len(label),np.float32)
    for index in range(left,right):
        weight[index] = 1.0

    if np.sum(weight) != 0:
        weight = weight / np.sum(weight)

    return weight

def trans_video_youtube(datasplit):

#     train_j = json.load(open(ANNOTATION_TRAIN_PARSED_PATH))
#     val_j = json.load(open(ANNOTATION_TEST_PARSED_PATH))
    #merge_j dictionary created previously only
#     merge_j=dict(train_j.items()+val_j.items())

    List = open(CONT_DIRECTORY+datasplit+'.txt').read().split('\n')[:-1] #get a list of h5 file, each file is a minibatch

    initial = 0
    cnt = 0
    fname = []
    title = []
    data = []
    label = []
    timestamps = []
    duration = []
    norm_timestamps = []
    segWeight = []
    
    for ele in List:
        print(ele)
        print(initial)
        train_batch = h5py.File(ele)
        for idx, video in enumerate(train_batch['title']):
            video = video.decode('utf-8')
            if video in merge_j.keys():
                for capidx, caption in enumerate(merge_j[video]['sentences']):
                    if len(caption.split(' ')) < 35:
                        fname.append(video)
                        duration.append(merge_j[video]['duration']) 
                        timestamps.append( merge_j[video]['timestamps'][capidx] )
                        norm_stamps = [merge_j[video]['timestamps'][capidx][0]/merge_j[video]['duration'], merge_j[video]['timestamps'][capidx][1]/merge_j[video]['duration']]
                        norm_timestamps.append(norm_stamps)
                        # title.append(unicodedata.normalize('NFKD', caption).encode('ascii','ignore'))
                        title.append(caption)
                        data.append(train_batch['data'][:,idx,:]) #insert item shape is (n_length,dim), so the data's shape will be (n_x,n_length,dim), so it need transpose
                        label.append(train_batch['label'][:,idx])
                        weights = getSegWeight(merge_j[video]['duration'], merge_j[video]['timestamps'][capidx], train_batch['label'][:,idx])
                        segWeight.append(weights)
                        cnt += 1 #sentence is enough for BATCH_SIZE
                        if cnt == BATCH_SIZE:
                            print(path+CONT_CAPTIONS_DIRECTORY_NAME+'/'+datasplit+str(initial)+'.h5')
                            batch = h5py.File(path+CONT_CAPTIONS_DIRECTORY_NAME+'/'+datasplit+str(initial)+'.h5','w')
                            data = np.transpose(data,(1,0,2))
                            batch['data'] = np.array(data)#np.zeros((n_length,BATCH_SIZE,4096*2))
                            fname = [a.encode('utf-8') for a in fname]
                            title = [a.encode('utf-8') for a in title]
                            fname = np.array(fname)
                            title = np.array(title)
                            batch['duration'] = duration
                            batch['fname'] = fname
                            batch['title'] = title
                            batch['timestamps'] = timestamps
                            batch['norm_timestamps'] = norm_timestamps
                            batch['weights'] = np.transpose(np.array(segWeight))
                            batch['label'] = np.transpose(np.array(label)) #np.zeros((n_length,BATCH_SIZE))
                            fname = []
                            duration = []
                            timestamps = []
                            norm_timestamps = []
                            title = []
                            label = []
                            segWeight = []
                            data = []
                            cnt = 0
                            initial += 1
        if ele == List[-1] and len(fname) > 0:
            while len(fname) < BATCH_SIZE:
                fname.append('')
                title.append('')
                timestamps.append([-1,-1])
                norm_timestamps.append([-1,-1])
                duration.append(-1)
            batch = h5py.File(path+CONT_CAPTIONS_DIRECTORY_NAME+'/'+datasplit+str(initial)+'.h5','w')
            batch['data'] = np.zeros((n_length,BATCH_SIZE,video_fts_dim))
            batch['data'][:,:len(data),:] = np.transpose(np.array(data),(1,0,2))#np.zeros((n_length,BATCH_SIZE,4096+1024))
            fname = [a.encode('utf-8') for a in fname]
            fname = np.array(fname)
            title = [a.encode('utf-8') for a in title]
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['duration'] = duration
            batch['timestamps'] = timestamps
            batch['norm_timestamps'] = norm_timestamps
            batch['weights'] = np.zeros((n_length,BATCH_SIZE))
            batch['weights'][:,:len(data)] = np.array(segWeight).T
            batch['label'] = np.ones((n_length,BATCH_SIZE))*(-1)
            batch['label'][:,:len(data)] = np.array(label).T



def getlist(CONT_CAPTIONS_DIRECTORY_NAME, split):
    list_path = os.path.join(CONT_CAPTIONS_DIRECTORY)
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')






# In[16]:


# if __name__ == '__main__':
if not os.path.exists(CONT_CAPTIONS_DIRECTORY):
    os.makedirs(CONT_CAPTIONS_DIRECTORY)
trans_video_youtube('train')
trans_video_youtube('val')
getlist(CONT_CAPTIONS_DIRECTORY_NAME,'train')
getlist(CONT_CAPTIONS_DIRECTORY_NAME,'val')
