import numpy as np
import os, h5py
import tqdm
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *


def getlist(Dir):
    f = h5py.File(AUDIO_FEATURES_H5PY_PATH, 'w')   
    for file in tqdm.tqdm(os.listdir(Dir)):
        f[file.split('.')[0]] = np.load(os.path.join(Dir,file))

getlist(os.path.join(AUDIO_FEATURES_DIRECTORY, AUDIO_FEATURES_TO_USE,""))



