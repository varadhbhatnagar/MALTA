#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import os, json, h5py, math, pdb, glob
import pickle
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *

MAX_LEN = 128
BATCH_SIZE = 20

all_c3d_fts = h5py.File(VIDEO_FEATURES_H5PY_PATH)

audio_data = h5py.File(AUDIO_FEATURES_H5PY_PATH)


if USE_FULL_SPLIT:
    splitdataset_path = FULL_SPLIT_DATASET_PATH
else:
    splitdataset_path = SMALL_SPLIT_DATASET_PATH
# In[19]:


# clip_sentence_pairs_iou = pickle.load(open("/home/mayur/exp/mount_swara/charades/ref_info/charades_sta_train_semantic_sentence_VP_sub_obj.pkl"))
# num_videos = len(clip_sentence_pairs_iou)  # 5182
# print(num_videos)
# # get the number of self.clip_sentence_pairs_iou
# clip_sentence_pairs_iou_all = []
# for ii in clip_sentence_pairs_iou:
#     print(ii)
#     for iii in clip_sentence_pairs_iou[ii]:
#         print(iii)
#         for iiii in range(len(clip_sentence_pairs_iou[ii][iii])):
#             print(iiii)
#             clip_sentence_pairs_iou_all.append(clip_sentence_pairs_iou[ii][iii][iiii])
#     break
# # print(clip_sentence_pairs_iou_all)


# In[ ]:





# In[20]:


movie_length_dict={}
with open(MOVIE_LENGTH_FILE_PATH)  as f:
    for l in f:
        content=l.rstrip().split(" ")
        content[1]=float(content[1])
        # content[2]=int(content[2])
        movie_length_dict[content[0]] = content[1]


# In[21]:


def get_c3d(video_name):
    #print(video_name, all_c3d_fts.keys())
    return np.array(all_c3d_fts[video_name]['c3d_features'])
    


# In[22]:


# a=get_c3d("Mystery_of_Krishnas_Siphon__Marathi.mp4")
# print(a.shape)


# In[23]:



# a.shape


# In[24]:


def get_max_len(path):
    lst = []
    for root, dirs, files in os.walk(path):
        for ele in files:
            if ele.endswith('json'):
                lst.append(root+'/'+ele)
    #print lst
    cnt = []
    for ele in lst:
        a = json.load(open(ele))
        cnt.append(len(a[0]))    
    return max(cnt)     





def get_VGG(f_path,ftype):
    if not os.path.exists(f_path + '.npz'):
        return []
    v = np.load(f_path + '.npz')[ftype]
    v=np.array(v)
    return v


def check_HL_nonHL_exist(label):
    idx = len(np.where(label == 1)[0])
    idy = len(np.where(label == 0)[0])
    return idx > 0 and idy > 0


def generate_h5py(X, y, q, fname, dataset, feature_folder_name, batch_start = 0):
    dirname = os.path.join(PROCESSED_OUTPUT_DIRECTORY, feature_folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    num = len(np.unique(q)) # how much videos here
    if num % BATCH_SIZE == 0:
        batch_num = int(num / BATCH_SIZE)
    else:
        batch_num = int(num / BATCH_SIZE) + 1
    q_idx = 0
    f_txt = open(os.path.join(dirname, dataset + '.txt'), 'w')
 
    for i in range(batch_start, batch_start + batch_num): # every h5 file contains BATCH_SIZE videos
        train_filename = os.path.join(dirname, dataset + str(i) + '.h5') #each file is a mini-batch
        
        print(train_filename)
        if os.path.isfile(train_filename):
            q_idx += BATCH_SIZE
            continue
        with h5py.File(train_filename, 'w') as f:
            f['data'] = np.zeros([MAX_LEN,BATCH_SIZE,X.shape[1]])  # a batch of features
            #f['label'] = np.zeros([MAX_LEN,BATCH_SIZE,2])
            f['label'] = np.zeros([MAX_LEN,BATCH_SIZE]) - 1  # all is -1
            f['cont'] = np.zeros([MAX_LEN,BATCH_SIZE])
            #f['reindex'] = np.zeros([MAX_LEN,BATCH_SIZE])
            f['reindex'] = np.zeros(MAX_LEN)
            fname_tmp = []
            title_tmp = []
            for j in range(BATCH_SIZE):
                X_id = np.where(q == q_idx)[0]  # find the video segment features of video q_idx
#                print(X_id)
                #while(len(X_id) == 0 or not check_HL_nonHL_exist(y[X_id])):
                while(len(X_id) == 0):
                    q_idx += 1
                    #print q_idx
                    X_id = np.where(q == q_idx)[0]
#                    print("QIDX , Q = ", str(q_idx), str(q))
                    if q_idx > max(q):
                        while len(fname_tmp) < BATCH_SIZE: #if the video is not enough for the last mini batch, insert ' '
                            fname_tmp.append('')
                            title_tmp.append('')
                        fname_tmp = [a.encode('utf-8') for a in fname_tmp]
                        title_tmp = [a.encode('utf-8') for a in title_tmp]
                        fname_tmp = np.array(fname_tmp)
                        title_tmp = np.array(title_tmp)
                        f['fname'] = fname_tmp
                        f['title'] = title_tmp
                        f_txt.write(train_filename + '\n')
                        return
                f['data'][:len(X_id),j,:] = X[X_id,:]
                print(f['data'].shape)
                f['label'][:len(X_id),j] = y[X_id]
                f['cont'][1:len(X_id)+1,j] = 1
                f['reindex'][:len(X_id)] = np.arange(len(X_id))
                f['reindex'][len(X_id):] = len(X_id)
                fname_tmp.append(fname[q_idx])
                title_tmp.append(fname[q_idx])
                if q_idx == q[-1]:
                    while len(fname_tmp) < BATCH_SIZE:
                        fname_tmp.append('')
                        title_tmp.append('')
                    fname_tmp = [a.encode('utf-8') for a in fname_tmp]
                    title_tmp = [a.encode('utf-8') for a in title_tmp]
                    fname_tmp = np.array(fname_tmp)
                    title_tmp = np.array(title_tmp)
                    f['fname'] = fname_tmp
                    f['title'] = title_tmp
                    f_txt.write(train_filename + '\n')
                    return
                q_idx += 1
                #print q_idx
            fname_tmp = [a.encode('utf-8') for a in fname_tmp]
            title_tmp = [a.encode('utf-8') for a in title_tmp]
            fname_tmp = np.array(fname_tmp)
            title_tmp = np.array(title_tmp)
            f['fname'] = fname_tmp.astype('S')
            f['title'] = title_tmp.astype('S')
        f_txt.write(train_filename + '\n')


def get_feats_depend_on_label(label, per_f, v, a, idx, d):
    #get features in one video, the feature is represented as clip features
    #label means the video clip division
    #perf: 1(VGG) 16(C3D)
    #C3D 16 frames a feature, and VGG 1 frames a feature
    X = []  # feature
    y = []  # indicate if video is finished
    q = []  # idx is the index of video in train/test/val dataset, all the segment in video will be tagged as idx in list q
#     print len(label[0])
    for l_index in range(len(label[0])):
        low = int(label[0][l_index][0])
        up = int(label[0][l_index][1])+1
        lowa = int(label[1][l_index][0])
        upa = int(label[1][l_index][1])+1
        #pdb.set_trace()
        if  low >= len(v) or lowa >= len(a) or low == up or lowa == upa:
            X.append(X[-1])
        else:
            
            X.append(np.concatenate((np.mean(v[low:up,:],axis=0), np.mean(a[lowa:upa,:],axis=0)),axis = 0))
           
        y.append(label[2][l_index])
        q.append(idx)
    return X, y, q


def load_feats(files, dataset, feature):
    #files: a list of video names
    #dataset: 'train' 'val' or 'test'
    #feature: 'c3d' 'VGG' 'cont'(conconation of c3d and VGG)
   
    X = [] # feature
    y = [] # indicate if video is finished
    q = [] # the index of video in train/test/val dataset
    fname = []
    idx = 0 # video index in the dataset 
    
    for ele in files: 
        print(ele, idx)
        l_path = os.path.join(LABEL_DIRECTORY, ele.split(".")[0] + '.json')
#         print l_path
        label = json.load(open(l_path))
        if len(label[0]) > MAX_LEN:
            continue

        f_path = ''
        if feature == 'c3d':
            v = get_c3d(ele)
            # alist = list(ele)
            # alist[0] = 'a'
            # aname = "".join(alist)
            # print(ele)
            
            a = audio_data[ele.split(".")[0]]
            per_f = 8
            #print(ele)
            if len(v) == 0:
                continue
            [x_tmp, y_tmp, q_tmp] = get_feats_depend_on_label(label, per_f, v, a, idx, dataset)
        elif feature == 'VGG':
            v = get_VGG(f_path,'fc7')
            per_f = 1
            if len(v) == 0:
                continue
            [x_tmp, y_tmp, q_tmp] = get_feats_depend_on_label(label, per_f, v, idx)
        elif feature == 'cont':
            v1 = get_c3d(ele)
            per_f1 = 8
            v2 = get_VGG(f_path,'fc7')
            per_f2 = 1
            if len(v1) == 0 or len(v2) == 0:
                continue
            [x1_tmp, y1_tmp, q1_tmp] = get_feats_depend_on_label(label, per_f1, v1, idx)
            [x2_tmp, y2_tmp, q2_tmp] = get_feats_depend_on_label(label, per_f2, v2, idx)
            x_tmp = map(list, zip(*(zip(*x1_tmp) + zip(*x2_tmp))))
            y_tmp = y1_tmp
            q_tmp = q1_tmp
        X += x_tmp
        y += y_tmp
        q += q_tmp
        #pdb.set_trace()
        fname.append(ele)
        idx += 1
    return np.array(X), np.array(y), np.array(q), np.array(fname)


def Normalize(X, normal = 0):
    if normal == 0:
        mean = np.mean(X,axis = 0)
        std = np.std(X,axis = 0)
        idx = np.where(std == 0)[0]
        std[idx] = 1
    else:
        mean = normal[0]
        std = normal[1]
    X = (X - mean) / std
    return X, mean, std



    




# In[25]:


def driver(inp_type, Rep_type, outp_folder_name):
    dataset = 'train'
    List = np.load(splitdataset_path)[dataset]#[0:1300] # get the train,val or test training video name
    # print(List[0:10])
    num=2

    for iii in range(int(math.ceil(len(List) / num*1.0))):
        [X, y, Q, fname] = load_feats(List[iii*num:min(len(List),(iii+1)*num)], dataset, Rep_type)
#         print X.shape
#         print y.shape
#         print Q.shape
#         print fname
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*10)
            
    dataset = 'val'
    List = np.load(splitdataset_path)[dataset]#[0:800]
    #print List[0:10]
    for iii in range(int(math.ceil(len(List) / num*1.0))):
        [X, y, Q, fname] = load_feats(List[iii*num:min(len(List),(iii+1)*num)], dataset, Rep_type)
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*10)
    
        

def getlist(path, split):
    List = glob.glob(path+split+'*.h5')
    print (path+split+'.txt')
    f = open(path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


# In[26]:


# if __name__ == '__main__':

driver('h5py', VIDEO_FEATURES_TO_USE, CONT_DIRECTORY_NAME)
getlist(CONT_DIRECTORY, 'train')
getlist(CONT_DIRECTORY, 'val')


# In[ ]:




