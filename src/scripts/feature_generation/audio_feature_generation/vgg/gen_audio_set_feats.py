from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf
from pathlib import Path
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
import tqdm
from os import path
import sys
sys.path.insert(0,'../../..')
from paths import *

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'audioset/vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

audio_path = PROCESSED_AUDIO_DIRECTORY
dest_path = VGG_AUDIO_FEATURES_DIRECTORY

Path(dest_path).mkdir(parents=True, exist_ok=True)

# max_frames = 20


with tf.device('/device:GPU:0'):
  def main(_):
    audio_files = os.listdir(audio_path)
    # maxi = 0
    for each_file in tqdm.tqdm(audio_files):
      file_nm = dest_path+each_file.split('.')[0]+'.npy'
      if not(path.exists(file_nm)):
        try:
          wav_file = audio_path+each_file
          examples_batch = vggish_input.wavfile_to_examples(wav_file)

          with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run([embedding_tensor],feed_dict={features_tensor: examples_batch})
            postprocessed_batch = embedding_batch
            #indices = np.linspace(0, len(postprocessed_batch), max_frames, endpoint=False, dtype=int)
            #postprocessed_batch = postprocessed_batch[indices]
            np.save(dest_path+each_file.split('.')[0]+'.npy',postprocessed_batch)
        except:
          print("here")
          continue

  if __name__ == '__main__':
    tf.app.run()
