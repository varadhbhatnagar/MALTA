import numpy as np
import os, json, h5py, math, pdb, glob


splitdataset_path_small = '/home/varad/Atma_Videos_New_Shortened/output/Atma_dataset_split_small.npz'

splitdataset_path_full = '/home/varad/Atma_Videos_New_Shortened/output/Atma_dataset_split_full.npz'

train_path = '/home/varad/Atma_Videos_New_Shortened/output/Annotation_video_mar_train_parsed.txt'
val_path = '/home/varad/Atma_Videos_New_Shortened/output/Annotation_video_mar_test_parsed.txt'


def generate_split_list():
    
    train_set=set({})
    i=0
    with open(train_path,"r") as f:
        for line in f:
#             print line
            content=line.rstrip().split(" ")
            train_set.add(content[0])
            i=i+1
#             if(i==10):
# #                 break
    train_list=list(train_set) 
#     print train_list
    val_set=set({})
    with open(val_path,"r") as f:
        for line in f:
#             print line
            content=line.rstrip().split(" ")
            val_set.add(content[0])
            i=i+1
#             if(i==20):
#                 break
    val_list=list(val_set)
#     print val_list
#     train_list = json.load(open(train_path)).keys()
#     val_list = json.load(open(val_path)).keys()

#     print len(train_list)
#     print len(val_list)

    all_train_num = len(train_list)
    half_train_num = int(0.5*all_train_num)

    train_list_small = train_list[:half_train_num]
    train_list_full = train_list

#     print train_list_small
#     print val_list

    print len(train_list_full)
    print len(train_list_small)
    print len(val_list)

    np.savez(splitdataset_path_full,train = train_list_full, val = val_list)
    np.savez(splitdataset_path_small,train = train_list_small, val = val_list)


if __name__ == '__main__':
    generate_split_list()



