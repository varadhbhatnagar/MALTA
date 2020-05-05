# All paths used in the project are defined in this file

import os

# General Paths

PROJECT_DIRECTORY   =  '/home/varad/MALTA_Restructured/'
DATA_DIRECTORY      =  os.path.join(PROJECT_DIRECTORY, "data", "")
SOURCE_DIRECTORY    =  os.path.join(PROJECT_DIRECTORY, "source", "")
MODEL_DIRECTORY     =  os.path.join(SOURCE_DIRECTORY, "model", "")
SCRIPTS_DIRECTORY   =  os.path.join(SOURCE_DIRECTORY, "scripts","")

# Data Paths : Add your Dataset paths here 

ATMA_RAW_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw", "Atma", "")
ATMA_RAW_VIDEOS_DIRECTORY = os.path.join(ATMA_RAW_DIRECTORY, "videos", "")
ATMA_RAW_CAPTIONS_DIRECTORY = os.path.join(ATMA_RAW_DIRECTORY, "captions", "")

ATMA_PROCESSED_DIRECTORY = os.path.join(DATA_DIRECTORY, "processed", "Atma", "")
ATMA_PROCESSED_VIDEOS_DIRECTORY = os.path.join(ATMA_PROCESSED_DIRECTORY, "videos", "")
ATMA_PROCESSED_CAPTIONS_DIRECTORY = os.path.join(ATMA_PROCESSED_DIRECTORY, "captions", "")
 

