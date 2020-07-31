import sys
import os
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *

dataset = 'Atma'
model_mode = 'AW'
task = 'train'
#model = ''
random_seed = 12
rnn_nlayers=1
dim_image_encode = -1
dim_image = 4096
dim_audio = 0
if AUDIO_FEATURES_TO_USE == 'mfcc':
    dim_audio = 40
else:
    dim_audio = 128
dim_word = 300
dim_hidden= 256
dim_hidden_video = 256
dim_hidden_audio = 64
dim_hidden_regress = 64
n_frame_step = 128
n_caption_step = 35
n_epochs =200
batch_size = 20
learning_rate = 0.001 
alpha_regress = 1.0
alpha_attention = 5.0
regress_layer_num = 2
iter_test = 10
iter_localize = 2
gpu_id = 3
rnn_dropout = 0.5
log_dir = '../../../logs/'
word_embedding_path = WORD_EMBEDDING_PATH
video_data_path_train = os.path.join(CONT_CAPTIONS_DIRECTORY, "train.txt")
video_data_path_val = os.path.join(CONT_CAPTIONS_DIRECTORY, "val.txt")
video_feat_path = CONT_CAPTIONS_DIRECTORY
model_save_dir = MODEL_SAVE_DIRECTORY
result_save_dir = RESULT_SAVE_DIRECTORY
words_dir = PROCESSED_OUTPUT_DIRECTORY
wordtoix_path = words_dir+'wordtoix.npy'
ixtoword_path = words_dir+'ixtoword.npy'
EMBEDDING_USED = 'fasttext'
word_fts_path = words_dir+'word_'+EMBEDDING_USED+'_fts_init.npy'
