import os
import codecs

input_path = '/home/varad/Atma_Videos_New_Shortened/output/'
annotation_folder = '/home/varad/Atma_Videos_New_Shortened/captions/'
output_path = '/home/varad/Atma_Videos_New_Shortened/output/'
annotation_video_mar_train = input_path+'Annotation_video_mar_train_path.txt'
annotation_video_mar_test = input_path+'Annotation_video_mar_test_path.txt'
annotation_video_mar_train_parsed = input_path+'Annotation_video_mar_train_parsed.txt'
annotation_video_mar_test_parsed = input_path+'Annotation_video_mar_test_parsed.txt'


def is_ascii(s):
    return True
#     try:
#         s.decode('ascii')
#         return True
#     except UnicodeDecodeError:
#         return False


def get_parsed_arr(filename):
    with open(filename, 'r', encoding = 'utf-8') as content_file:
        content = content_file.read()
        
    time_precision=4
    parsed_arr=[]
    colon_idx=content.find(':')
    colon_idx_2=content.find(':',colon_idx+1)
    colon_idx_3=content.find(':',colon_idx_2+1)

    if(colon_idx_3-colon_idx_2==3):
        time_precision=4
    else:
        time_precision=2

    if (time_precision==4):
        colon_idx=0
        while(content.find(':',colon_idx+1)!=-1):
            curr_anno={}
 
            while(True):
                colon_idx=content.find(':',colon_idx)
                if(ord(content[colon_idx+1])-ord('0')>=0 and ord(content[colon_idx+1])-ord('0')<=9):
                    break
                else:
                    colon_idx = colon_idx+1
            
            colon_idx=colon_idx-2
            s_hh,s_mm,s_ss,s_ms=content[colon_idx:colon_idx+12].split(":")
#             print(s_hh,s_mm,s_ss,s_ms)
            s_time=int(s_mm)*60+int(s_ss)
            s_time=str(s_time)+"."+str(s_ms[0])
            colon_idx=colon_idx+13
#             colon_idx
#             if(colon_idx>=len(content)):
#                 break
            if(is_ascii(content[colon_idx])==True):
                s_str_idx=colon_idx
            else:
                s_str_idx=colon_idx+1
#             print(content[s_str_idx]=='\xc2')
            
            while(True):
                colon_idx=content.find(":",colon_idx)
                if(ord(content[colon_idx+1])-ord('0')>=0 and ord(content[colon_idx+1])-ord('0')<=9):
                    break
                else:
                    colon_idx = colon_idx+1
            
            colon_idx=colon_idx-2
            e_str_idx=colon_idx
#             print(content[s_str_idx:e_str_idx].rstrip())
            caption=content[s_str_idx:e_str_idx].rstrip().replace("\n","")
            e_hh,e_mm,e_ss,e_ms=content[colon_idx:colon_idx+12].split(":")
#             print(e_hh,e_mm,e_ss,e_ms)
            e_time=int(e_mm)*60+int(e_ss)
            e_time=str(e_time)+"."+str(e_ms[0])

            curr_anno['start']=s_time
            curr_anno['end']=e_time
            curr_anno['caption']=caption
            parsed_arr.append(curr_anno)
            colon_idx=colon_idx+12
    else:
        colon_idx=0
        while(content.find(':',colon_idx+1)!=-1):
            curr_anno={}
            
            while(True):
                colon_idx=content.find(":",colon_idx)
                if(ord(content[colon_idx+1])-ord('0')>=0 and ord(content[colon_idx+1])-ord('0')<=9):
                    break
                else:
                    colon_idx = colon_idx+1

            colon_idx=colon_idx-2
            s_mm,s_ss=content[colon_idx:colon_idx+5].split(":")
            s_ms='000'
#             print(s_mm,s_ss,s_ms)
            s_time=int(s_mm)*60+int(s_ss)
            s_time=str(s_time)+"."+str(s_ms[0])
            colon_idx=colon_idx+6
#             colon_idx
            if(is_ascii(content[colon_idx])==True):
                s_str_idx=colon_idx
            else:
                s_str_idx=colon_idx+1
#             print(content[s_str_idx]=='\xc2')
            
            while(True):
                colon_idx=content.find(":",colon_idx)
                if(ord(content[colon_idx+1])-ord('0')>=0 and ord(content[colon_idx+1])-ord('0')<=9):
                    break
                else:
                    colon_idx = colon_idx+1

            colon_idx=colon_idx-2
            e_str_idx=colon_idx
#             print(content[s_str_idx:e_str_idx].rstrip())
            caption=content[s_str_idx:e_str_idx].rstrip().replace("\n","")
            e_mm,e_ss=content[colon_idx:colon_idx+5].split(":")
            e_ms='000'
#             print(e_mm,e_ss,e_ms)
            e_time=int(e_mm)*60+int(e_ss)
            e_time=str(e_time)+"."+str(e_ms[0])
            curr_anno['start']=s_time
            curr_anno['end']=e_time
            curr_anno['caption']=caption
            parsed_arr.append(curr_anno)
            colon_idx=colon_idx+5
    return parsed_arr
#             break
#     else:
        

annotation_video_mar_train_parsed_file=open(annotation_video_mar_train_parsed,"w")
count_idx=0
with open(annotation_video_mar_train,"r", encoding = 'utf-8') as f:
    for line in f:
        count_idx=count_idx+1
        if(count_idx%20==0):
            print(count_idx)
        video_file, anno_file=line.rstrip().split(",")
        try:
            anno_arr=get_parsed_arr(anno_file)
        except:
            print(anno_file)
        video_id=video_file.split("/")[-1]
        for i in range(0,len(anno_arr)):
            anno_arr[i]['caption']=anno_arr[i]['caption'].replace(",","")
            anno_arr[i]['caption']=anno_arr[i]['caption'].replace("-"," ")
            anno_arr[i]['caption']=anno_arr[i]['caption'].replace("!","")
            
            annotation_video_mar_train_parsed_file.write(video_id+" "+anno_arr[i]['start']+" "+anno_arr[i]['end']+"##"+anno_arr[i]['caption'].rstrip()+"\n")
#         break
annotation_video_mar_train_parsed_file.close()

annotation_video_mar_test_parsed_file=open(annotation_video_mar_test_parsed,"w")
count_idx=0
with open(annotation_video_mar_test,"r", encoding = 'utf-8') as f:
    for line in f:
        count_idx=count_idx+1
        if(count_idx%20==0):
            print(count_idx)
        video_file, anno_file=line.rstrip().split(",")
        try:
            anno_arr=get_parsed_arr(anno_file)
        except:
            print(anno_file)
        video_id=video_file.split("/")[-1]
        for i in range(0,len(anno_arr)):
            anno_arr[i]['caption']=anno_arr[i]['caption'].replace(",","")
            anno_arr[i]['caption']=anno_arr[i]['caption'].replace("-"," ")
            anno_arr[i]['caption']=anno_arr[i]['caption'].replace("!","")
            annotation_video_mar_test_parsed_file.write(video_id+" "+anno_arr[i]['start']+" "+anno_arr[i]['end']+"##"+anno_arr[i]['caption'].rstrip()+"\n")
#         break
annotation_video_mar_test_parsed_file.close()

