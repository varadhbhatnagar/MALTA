import numpy as np
import os, json, h5py, math, pdb, glob
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *


def generate_split_list():
    
    train_set=set({})
    i=0
    with open(ANNOTATION_TRAIN_PARSED_PATH,"r") as f:
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
    with open(ANNOTATION_TEST_PARSED_PATH,"r") as f:
        for line in f:
#             print line
            content=line.rstrip().split(" ")
            val_set.add(content[0])
            i=i+1
#             if(i==20):
#                 break
    val_list=list(val_set)
#     print val_list
#     train_list = json.load(open(ANNOTATION_TRAIN_PARSED_PATH)).keys()
#     val_list = json.load(open(ANNOTATION_TEST_PARSED_PATH)).keys()

#     print len(train_list)
#     print len(val_list)

    all_train_num = len(train_list)
    half_train_num = int(0.5*all_train_num)

    train_list_small = train_list[:half_train_num]
    train_list_full = train_list

#     print train_list_small
#     print val_list

    print(len(train_list_full))
    print(len(train_list_small))
    print(len(val_list))

    np.savez(FULL_SPLIT_DATASET_PATH,train = train_list_full, val = val_list)
    np.savez(SMALL_SPLIT_DATASET_PATH,train = train_list_small, val = val_list)


if __name__ == '__main__':
    generate_split_list()



