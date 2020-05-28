import os
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *


with open(VIDEO_LIST_PATH, "w") as output:
    for file in os.listdir(PROCESSED_VIDEOS_DIRECTORY):
        if file.endswith(VIDEO_EXTENSION):
            li = os.path.join("", file)
            output.write(str(li))
            output.write("\n")
