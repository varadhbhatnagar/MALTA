import os
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *

with open(ANNOTATION_LIST_PATH, "w") as output:
    for file in os.listdir(PROCESSED_CAPTIONS_DIRECTORY):
        if file.endswith(CAPTION_EXTENSION):
            li = os.path.join("", file)
            output.write(str(li))
            output.write("\n")
