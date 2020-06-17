from python_speech_features import mfcc, delta, logfbank, ssc
import scipy.io.wavfile as wav
import os
from pathlib import Path
import numpy as np
from sklearn import preprocessing
import sys
sys.path.insert(0,'../../..')
from paths import *


def calculate_delta(array):
    """Calculate and returns the delta of given feature vector matrix"""

    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas

Path(MFCC_AUDIO_FEATURES_DIRECTORY).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(PROCESSED_AUDIO_DIRECTORY):
			(rate, sig) = wav.read(PROCESSED_AUDIO_DIRECTORY+filename)
			mfcc_feature = mfcc(sig,rate,0.025, 0.01,20, appendEnergy = True)
			print(mfcc_feature.shape)
			mfcc_feature = preprocessing.scale(mfcc_feature)
			print(mfcc_feature.shape)
			delta = calculate_delta(mfcc_feature)
			combined = np.hstack((mfcc_feature,delta)) 
			print(combined.shape)
			np.save(MFCC_AUDIO_FEATURES_DIRECTORY+filename.split('.')[0],combined)
			print(filename+" Done")



