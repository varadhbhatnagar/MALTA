# Constants used in the project are defined in this file
from paths import *

VIDEO_EXTENSION = ".mp4"	# Include '.' before the extension
AUDIO_EXTENSION = "wav"		# Do not include '.' before the extension
CAPTION_EXTENSION = ".txt"	# Include '.' before the extension
CAPTION_LANGUAGE = "mar"        #  mar for Marathi
                                #  eng for English
AUDIO_FEATURES_TO_USE = "vgg"
TRAIN_TEST_SPLIT = 0.8

VIDEO_LIST_FILENAME = "video_list.txt"
C3D_FEATURES_FILENAME = "c3d_features.hdf5"
ANNOTATION_PATH_FILENAME = "annotation_path.txt"
ANNOTATION_TRAIN_PATHS_FILENAME = "Annotation_video_"+CAPTION_LANGUAGE+"_train_path.txt"
ANNOTATION_TEST_PATHS_FILENAME = "Annotation_video_" +CAPTION_LANGUAGE+"_test_path.txt"
ERROR_FILENAME = "error.txt"
ANNOTATION_TRAIN_PARSED_FILENAME = "Annotation_video_"+CAPTION_LANGUAGE+"_train_parsed.txt"
ANNOTATION_TEST_PARSED_FILENAME = "Annotation_video_"+CAPTION_LANGUAGE+"_test_parsed.txt"
MOVIE_LENGTH_FILENAME = "movie_length.txt"
LABEL_FOLDERNAME="middle-labels"
NUMBER_OF_SEGMENTS=128
