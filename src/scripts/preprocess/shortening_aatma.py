import os
import re 
import datetime
from pathlib import Path
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import sys
sys.path.insert(0,'..')
from paths import *


timestamp_regex = re.compile(r'\d:\d\d:\d\d.\d\d\d')


def convert_to_miliseconds(ts):
    return (int(ts[0:1])*60*60 + int(ts[2:4])*60+ int(ts[5:7])) * 1000 + int(ts[8:11])


def findreplace(char, string):
   return ''.join(string.split(char))


Path(ATMA_PROCESSED_VIDEOS_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path(ATMA_PROCESSED_CAPTIONS_DIRECTORY).mkdir(parents=True, exist_ok=True)

for caption_filename in os.listdir(ATMA_RAW_CAPTIONS_DIRECTORY):
    video_cut_ts = []
    print(caption_filename)

    with open(ATMA_RAW_CAPTIONS_DIRECTORY+ caption_filename,'r') as file:
        data = file.read()
        original_timestamps = timestamp_regex.findall(data)
        original_timestamps.sort()
        converted_timestamps = []

        for i in range(len(original_timestamps)):
            converted_timestamps.append(convert_to_miliseconds(original_timestamps[i]))

        prev_timestamp = 0
        current_timestamp = 2 * 60 * 1000
       
        for i in range(0,len(converted_timestamps)-1,2):
            
            if(converted_timestamps[i] <= current_timestamp and current_timestamp <= converted_timestamps[i+1]):
                video_cut_ts.append(prev_timestamp)
                video_cut_ts.append(current_timestamp)
                prev_timestamp = converted_timestamps[i]
                current_timestamp = current_timestamp + 2* 60 * 1000
                i = i -2

            elif(current_timestamp <= converted_timestamps[i]):
                video_cut_ts.append(prev_timestamp)
                video_cut_ts.append(current_timestamp)
                prev_timestamp = current_timestamp
                current_timestamp = current_timestamp + 2* 60 * 1000
        
        video_cut_ts.append(prev_timestamp)
        video_cut_ts.append(current_timestamp)

        print(video_cut_ts)

    with open(ATMA_RAW_CAPTIONS_DIRECTORY+ caption_filename,'r') as file:
        data = file.read()

        #print(original_timestamps)
        #print(converted_timestamps)
        
        video_names = os.listdir(ATMA_RAW_VIDEOS_DIRECTORY)
        video_filename = findreplace('-', caption_filename).split('.')[0]
        video_filename = video_filename[:len(video_filename)-3]
        VIDEO_FOUND = 0

        for vid in video_names:
            if vid.lower() == video_filename.lower()+'.mp4':
                corresponding_video = vid
                VIDEO_FOUND = 1
                break
        if VIDEO_FOUND == 0:
            print("Not Found")
        z = 1
        for i in range(0, len(video_cut_ts)-1, 2):
            modified_caption_filename = ATMA_PROCESSED_CAPTIONS_DIRECTORY+caption_filename.split('.')[0]+str(z)+'.txt'
            cf = open(modified_caption_filename, "w")

            for k in range(0,len(converted_timestamps),2):
                
                #print(converted_timestamps[k], converted_timestamps[k+1], video_cut_ts[i], video_cut_ts[i+1])
                if converted_timestamps[k]>=video_cut_ts[i] and converted_timestamps[k+1]<=video_cut_ts[i+1]:
                    start_index = data.find(original_timestamps[k+1])
                    if k == len(converted_timestamps)-2:
                        end_index = len(data)-2
                    else: 
                        end_index = data.find(original_timestamps[k+2])-2
                    #print("Converted TS +"+ str(converted_timestamps[k]-video_cut_ts[i]))
                    sts = ("0"+str(datetime.timedelta(milliseconds = converted_timestamps[k]-video_cut_ts[i])))
                    if len(sts) == 8:
                        sts = sts[0:8]+":000"+" "
                    else:
                        sts = sts[0:8]+":"+sts[9:12]+" "
                    
                    cf.write(sts)
                    cf.write(data[start_index+12:end_index]+' ')
                    ets = ("0"+str(datetime.timedelta(milliseconds =converted_timestamps[k+1]-video_cut_ts[i])))
                    if len(ets) == 8:
                        ets = ets[0:8] +":000" + "\n"
                    else:
                        ets = ets[0:8]+":"+ets[9:12]+"\n"
                    cf.write(ets)
                    
            cf.close()        

            if os.stat(modified_caption_filename).st_size != 0:
                #print(video_filename)
                ffmpeg_extract_subclip(ATMA_RAW_VIDEOS_DIRECTORY+corresponding_video, video_cut_ts[i]/1000, video_cut_ts[i+1]/1000, targetname=ATMA_PROCESSED_VIDEOS_DIRECTORY+ corresponding_video.split('.')[0]+str(z)+'.mp4')

            else:
                os.remove(modified_caption_filename)
                
            z = z + 1
            
            
