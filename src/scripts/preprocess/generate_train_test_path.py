import os
import random
import math
import re
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *

video_list=[]
annotation_list=[]
video_list = os.listdir(PROCESSED_VIDEOS_DIRECTORY)

with open(ANNOTATION_LIST_PATH) as f:
    for line in f:
        annotation_list.append(line.rstrip())


annotation_video_arr=[]

err_arr=[]
count=0

for a_file in annotation_list:
    
    file_whole_name=a_file
    file_name=file_whole_name.split(".")[0]
    file_name=file_name.replace("-","")
    file_name=file_name.replace("_","")

    lang=file_name.split(" ")[-1]
    
    count=count+1
    file_found=0

    for a_video in video_list:
        a_video_ext = a_video.split(".")[1]
        a_video = a_video.split(".")[0]
	#if file_name.lower().find(a_video.lower()) == 0:				For normal AATMA
        if re.sub(r'MAR([0-9]+)',r'\1', file_name).lower().find(a_video.lower()) == 0:		# For shortened AATMA
            annotation_video_arr.append(PROCESSED_VIDEOS_DIRECTORY+a_video+"."+a_video_ext+","+PROCESSED_CAPTIONS_DIRECTORY+a_file)
            file_found = 1
            break

    if(file_found==0):
        err_arr.append(a_file)

random.shuffle(annotation_video_arr)
annotation_video_train_arr=annotation_video_arr[:int(math.ceil(len(annotation_video_arr)*(TRAIN_TEST_SPLIT)))]
annotation_video_test_arr=annotation_video_arr[int(math.ceil(len(annotation_video_arr)*(TRAIN_TEST_SPLIT))):]

with open(ANNOTATION_TRAIN_PATH, 'w') as f:
    for item in annotation_video_train_arr:
        f.write("%s\n" % item)

with open(ANNOTATION_TEST_PATH, 'w') as f:
    for item in annotation_video_test_arr:
        f.write("%s\n" % item)

with open(ERROR_FILE_PATH, 'w') as f:
    for item in err_arr:
        f.write("%s\n" % item)

