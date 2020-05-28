# All paths used in the project are defined in this file

import os
from constants import *

# Dataset to be used

DATASET_NAME = "Atma"

# General Paths

PROJECT_DIRECTORY   =  '/home/varad/MALTA_Restructured/'
DATA_DIRECTORY      =  os.path.join(PROJECT_DIRECTORY,"data", "")
SOURCE_DIRECTORY    =  os.path.join(PROJECT_DIRECTORY, "source", "")
MODEL_DIRECTORY     =  os.path.join(SOURCE_DIRECTORY, "model", "")
SCRIPTS_DIRECTORY   =  os.path.join(SOURCE_DIRECTORY, "scripts","")
FEATURES_DIRECTORY  =  os.path.join(DATA_DIRECTORY, "features", "")

# Data Paths : Add your Dataset paths here 

# Paths for Atma Dataset
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw", DATASET_NAME, "")
RAW_VIDEOS_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "videos", "")
RAW_CAPTIONS_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "captions", "")

PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "processed", DATASET_NAME, "")
PROCESSED_VIDEOS_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "videos", "")
PROCESSED_CAPTIONS_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "captions", "")
PROCESSED_AUDIO_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "audio", "")
PROCESSED_OUTPUT_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "output", "")

# Paths for Features

AUDIO_FEATURES_DIRECTORY = os.path.join(FEATURES_DIRECTORY, DATASET_NAME, "audio_features", "")
VGG_AUDIO_FEATURES_DIRECTORY = os.path.join(AUDIO_FEATURES_DIRECTORY, "vgg","")
MFCC_AUDIO_FEATURES_DIRECTORY = os.path.join(AUDIO_FEATURES_DIRECTORY, "mfcc", "")

VIDEO_FEATURES_DIRECTORY = os.path.join(FEATURES_DIRECTORY, DATASET_NAME, "video_features")
C3D_VIDEO_FEATURES_DIRECTORY = os.path.join(VIDEO_FEATURES_DIRECTORY, "c3d")

#Paths for Output Files

VIDEO_LIST_PATH = os.path.join(PROCESSED_OUTPUT_DIRECTORY, VIDEO_LIST_FILENAME)
ANNOTATION_LIST_PATH = os.path.join(PROCESSED_OUTPUT_DIRECTORY, ANNOTATION_PATH_FILENAME)
AUDIO_FEATURES_H5PY_PATH = os.path.join(PROCESSED_OUTPUT_DIRECTORY, "audio_features_"+AUDIO_FEATURES_TO_USE+".h5py")
