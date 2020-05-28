import os
import sys
from pathlib import Path
sys.path.insert(0,'..')
from paths import *
from constants import *


EXTRACT_AUDIO_COMMAND = ('ffmpeg -i "{from_video_path}" '
                         '-f {audio_ext} '
                         '-vn "{to_audio_path}"')

Path(PROCESSED_AUDIO_DIRECTORY).mkdir(parents=True, exist_ok=True)

for filename in os.listdir(PROCESSED_VIDEOS_DIRECTORY):
	if not filename.endswith(VIDEO_EXTENSION):
        	continue

	audio_file_name = '{}.{}'.format(filename.split('.')[0], AUDIO_EXTENSION)
	command = EXTRACT_AUDIO_COMMAND.format(from_video_path=os.path.join(PROCESSED_VIDEOS_DIRECTORY, filename), audio_ext=AUDIO_EXTENSION, to_audio_path=os.path.join(PROCESSED_AUDIO_DIRECTORY, audio_file_name))
	os.system(command)
