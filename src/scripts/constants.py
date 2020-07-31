# Constants used in the project are defined in this file
from paths import *

VIDEO_EXTENSION = ".mp4"	# Include '.' before the extension
AUDIO_EXTENSION = "wav"		# Do not include '.' before the extension
CAPTION_EXTENSION = ".txt"	# Include '.' before the extension
CAPTION_LANGUAGE = "mar"        #  mar for Marathi
                                #  eng for English
AUDIO_FEATURES_TO_USE = "mfcc"  # mfcc or vgg
VIDEO_FEATURES_TO_USE = "c3d"
TRAIN_TEST_SPLIT = 0.8
CONT_CAPTION_FILE_FEATURE_DIMENSION_VGG = 4096+128
CONT_CAPTION_FILE_FEATURE_DIMENSION_MFCC = 4096+40
BATCH_SIZE = 20
VIDEO_LIST_FILENAME = "video_list.txt"
C3D_FEATURES_FILENAME = "c3d_features.hdf5"
ANNOTATION_PATH_FILENAME = "annotation_path.txt"
ANNOTATION_TRAIN_PATHS_FILENAME = "Annotation_video_"+CAPTION_LANGUAGE+"_train_path.txt"
ANNOTATION_TEST_PATHS_FILENAME = "Annotation_video_" +CAPTION_LANGUAGE+"_test_path.txt"
ERROR_FILENAME = "error.txt"
ANNOTATION_TRAIN_PARSED_FILENAME = "Annotation_video_"+CAPTION_LANGUAGE+"_train_parsed.txt"
ANNOTATION_TEST_PARSED_FILENAME = "Annotation_video_"+CAPTION_LANGUAGE+"_test_parsed.txt"
MOVIE_LENGTH_FILENAME = "movie_length.txt"
LABEL_FOLDERNAME="middle-labels"+"_"+AUDIO_FEATURES_TO_USE
NUMBER_OF_SEGMENTS=128
FULL_SPLIT_DATASET_FILENAME = "dataset_split_full.npz"
SMALL_SPLIT_DATASET_FILENAME = "dataset_split_small.npz"
USE_FULL_SPLIT = True
CONT_DIRECTORY_NAME = "video_audio_cont_" + AUDIO_FEATURES_TO_USE
CONT_CAPTIONS_DIRECTORY_NAME = "video_audio_cont_captions_" + AUDIO_FEATURES_TO_USE

