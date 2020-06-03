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

path = '/home/data/Atma_Videos_New_Shortened/c3d_features/h5py/'
feature_folder = 'video_audio_cont_captions'

train_captions_path = '/home/data/Atma_Videos_New_Shortened/output/Annotation_video_mar_train_parsed.txt'
val_captions_path = '/home/data/Atma_Videos_New_Shortened/output/Annotation_video_mar_test_parsed.txt'
movie_detail_path = '/home/data/Atma_Videos_New_Shortened/output/movie_length.txt'
splitdataset_path = '/home/data/Atma_Videos_New_Shortened/output/Atma_dataset_split_full.npz'

batch_size = 20 # each mini batch will have batch_size sentences
n_length = 128
video_fts_dim = 4096+40




# In[2]:


movie_length_dict={}
with open(movie_detail_path)  as f:
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
    with open(train_captions_path) as f:
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
                    train_j[content[0]]['sentences'].append(unicode(line.rstrip().split("##")[1], "utf-8"))
                else:
                    s_time=float(content[1])
                    e_time=float(content[2].split("##")[0])
                    time_arr=[]
                    time_arr.append(s_time)
                    time_arr.append(e_time)
                    train_j[content[0]]['timestamps'].append(time_arr)
                    train_j[content[0]]['sentences'].append(unicode(line.rstrip().split("##")[1], "utf-8"))
    val_j={}
    dataset = 'val'
    List = np.load(splitdataset_path)[dataset]#[0:800]
    with open(val_captions_path) as f:
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
                    val_j[content[0]]['sentences'].append(unicode(line.rstrip().split("##")[1], "utf-8"))
                else:
                    s_time=float(content[1])
                    e_time=float(content[2].split("##")[0])
                    time_arr=[]
                    time_arr.append(s_time)
                    time_arr.append(e_time)
                    val_j[content[0]]['timestamps'].append(time_arr)
                    val_j[content[0]]['sentences'].append(unicode(line.rstrip().split("##")[1], "utf-8"))
    z = train_j.copy()
    z.update(val_j)
    return z
    


# In[13]:


merge_j=create_merge_j_dict();


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

#     train_j = json.load(open(train_captions_path))
#     val_j = json.load(open(val_captions_path))
    #merge_j dictionary created previously only
#     merge_j=dict(train_j.items()+val_j.items())

    List = open(path+'/video_audio_cont_mfcc/'+datasplit+'.txt').read().split('\n')[:-1] #get a list of h5 file, each file is a minibatch

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
        print ele
        print initial
        train_batch = h5py.File(ele)
        for idx, video in enumerate(train_batch['title']):
            if video in merge_j.keys():
                for capidx, caption in enumerate(merge_j[video]['sentences']): 
                    if len(caption.split(' ')) < 35:
		            fname.append(video)
		            duration.append(merge_j[video]['duration']) 
		            timestamps.append( merge_j[video]['timestamps'][capidx] )
		            norm_stamps = [merge_j[video]['timestamps'][capidx][0]/merge_j[video]['duration'], merge_j[video]['timestamps'][capidx][1]/merge_j[video]['duration']]
		            norm_timestamps.append(norm_stamps)
		            # title.append(unicodedata.normalize('NFKD', caption).encode('ascii','ignore'))
		            title.append(caption.encode('utf-8'))
		            data.append(train_batch['data'][:,idx,:]) #insert item shape is (n_length,dim), so the data's shape will be (n_x,n_length,dim), so it need transpose
		            label.append(train_batch['label'][:,idx])
		            weights = getSegWeight(merge_j[video]['duration'], merge_j[video]['timestamps'][capidx], train_batch['label'][:,idx])
		            segWeight.append(weights)
		            cnt += 1 #sentence is enough for batch_size
		            
		            if cnt == batch_size:
		                print(path+feature_folder+'/'+datasplit+str(initial)+'.h5')
		                batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
		                data = np.transpose(data,(1,0,2))
		                batch['data'] = np.array(data)#np.zeros((n_length,batch_size,4096*2))
		                fname = np.array(fname)
		                title = np.array(title)
		                batch['duration'] = duration
		                batch['fname'] = fname
		                batch['title'] = title
		                batch['timestamps'] = timestamps
		                batch['norm_timestamps'] = norm_timestamps
		                batch['weights'] = np.transpose(np.array(segWeight))
		                batch['label'] = np.transpose(np.array(label)) #np.zeros((n_length,batch_size))
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
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
                timestamps.append([-1,-1])
                norm_timestamps.append([-1,-1])
                duration.append(-1)
            batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
            batch['data'] = np.zeros((n_length,batch_size,video_fts_dim))
            batch['data'][:,:len(data),:] = np.transpose(np.array(data),(1,0,2))#np.zeros((n_length,batch_size,4096+1024))
            fname = np.array(fname)
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['duration'] = duration
            batch['timestamps'] = timestamps
            batch['norm_timestamps'] = norm_timestamps
            batch['weights'] = np.zeros((n_length,batch_size))
            batch['weights'][:,:len(data)] = np.array(segWeight).T
            batch['label'] = np.ones((n_length,batch_size))*(-1)
            batch['label'][:,:len(data)] = np.array(label).T




def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')






# In[16]:


# if __name__ == '__main__':
if not os.path.exists(path+feature_folder):
    os.makedirs(path+feature_folder)
trans_video_youtube('train')
trans_video_youtube('val')
getlist(feature_folder,'train')
getlist(feature_folder,'val')
