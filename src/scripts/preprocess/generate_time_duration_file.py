import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *

def getLength(filename):
    result = subprocess.Popen(['ffprobe', filename], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, encoding="utf-8")
    return [x for x in result.stdout.readlines() if "Duration" in x]

video_list=[]
video_list = os.listdir(PROCESSED_VIDEOS_DIRECTORY)

movie_length_file=open(MOVIE_LENGTH_FILE_PATH,"w")
idx=0
for a_video in video_list:
    print(PROCESSED_VIDEOS_DIRECTORY+a_video)
    dur_string=getLength(PROCESSED_VIDEOS_DIRECTORY+a_video) 
    dur_substr=dur_string[0][12:23]
    print(dur_substr)
    hh_mm_ss, ms=dur_substr.split(".")
    hh,mm,ss=hh_mm_ss.split(":")
    total_dur=str(int(mm)*60+int(ss))+"."+ms
    movie_length_file.write(a_video+" "+total_dur+"\n")
    idx=idx+1
#     if(idx==10):
#         break
movie_length_file.close()
