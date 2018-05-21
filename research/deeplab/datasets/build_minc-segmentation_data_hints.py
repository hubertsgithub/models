# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
import os
import random
import string
import sys
import build_data
import tensorflow as tf
import numpy as np
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_image_folder',
    './minc-segmentation/images_preprocessed/train',
    'Folder containing training images')
tf.app.flags.DEFINE_string(
    'train_image_label_folder',
    './minc-segmentation/labels_preprocessed/train',
    'Folder containing annotations for training images')
tf.app.flags.DEFINE_string(
    'train_image_class_hint_folder',
    './minc-segmentation/class_hints_preprocessed/train',
    'Folder containing class hints for training images')

tf.app.flags.DEFINE_string(
    'val_image_folder',
    './minc-segmentation/images_preprocessed/val',
    'Folder containing val images')
tf.app.flags.DEFINE_string(
    'val_image_label_folder',
    './minc-segmentation/labels_preprocessed/val',
    'Folder containing annotations for val images')
tf.app.flags.DEFINE_string(
    'val_image_class_hint_folder',
    './minc-segmentation/class_hints_preprocessed/val',
    'Folder containing class hints for val images')

tf.app.flags.DEFINE_string(
    'test_image_folder',
    './minc-segmentation/images_preprocessed/test',
    'Folder containing test images')
tf.app.flags.DEFINE_string(
    'test_image_label_folder',
    './minc-segmentation/labels_preprocessed/test',
    'Folder containing annotations for test images')
tf.app.flags.DEFINE_string(
    'test_image_class_hint_folder',
    './minc-segmentation/class_hints_preprocessed/test',
    'Folder containing class hints for test images')

tf.app.flags.DEFINE_string(
    'output_dir', './minc-segmentation/tfrecord',
    'Path to save converted SSTable of Tensorflow example')

_NUM_SHARDS = 4

def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir, dataset_hint_dir=None, dummy_hints=False):
  """ Converts the minc-segmentation dataset into into tfrecord format (SSTable).

  Args:
    dataset_split: Dataset split (e.g., train, val).
    dataset_dir: Dir in which the dataset locates.
    dataset_label_dir: Dir in which the annotations locates.

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  hints = (dataset_hint_dir is not None)

  img_names = tf.gfile.Glob(os.path.join(dataset_dir, '*.jpg'))
  random.shuffle(img_names)
  seg_names = []
  if hints:
    hint_names = []
  for f in img_names:
    # get the filename without the extension
    basename = os.path.basename(f).split(".")[0]
    # cover its corresponding *_seg.png
    seg = os.path.join(dataset_label_dir, basename+'.png')
    seg_names.append(seg)
    if hints:
        hint = os.path.join(dataset_hint_dir, basename+'.png')
        hint_names.append(hint)

  num_images = len(img_names)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)
  if hints:
    hint_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, num_images, shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = img_names[i]
        image_data = tf.gfile.FastGFile(image_filename, 'r').read()
        height, width = image_reader.read_image_dims(image_data)

        # Read the semantic segmentation annotation.
        seg_filename = seg_names[i]
        seg_data = tf.gfile.FastGFile(seg_filename, 'r').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')

        if hints:
            if dummy_hints:
                temp_name = '/tmp/tempfile-{}.png'.format(os.getpid())
                if not os.path.exists(temp_name):
                    sys.stderr.write('\nCreating temp file {}'.format(temp_name))
                    sys.stderr.flush()
                    sys.stderr.write('\n***REMOVE IF CODE CRASHES***')
                    sys.stderr.flush()
                hint_data = 255*np.ones((height,width), dtype=np.uint8)
                hint_data = Image.fromarray(hint_data, mode='L')
                hint_data.save(temp_name)
                hint_data = tf.gfile.FastGFile(temp_name, 'r').read()
            else:
                hint_filename = hint_names[i]
                hint_data = tf.gfile.FastGFile(hint_filename, 'r').read()
            hint_height, hint_width = hint_reader.read_image_dims(hint_data)
            if height != hint_height or width != hint_width:
                raise RuntimeError('Shape mismatched between image and hints.')

        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data=image_data,
            filename=img_names[i],
            height=height,
            width=width,
            seg_data=seg_data,
            class_hint_data=hint_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def main(unused_argv):
  tf.gfile.MakeDirs(FLAGS.output_dir)

  _convert_dataset('train_class_hints',
                   dataset_dir=FLAGS.train_image_folder,
                   dataset_label_dir=FLAGS.train_image_label_folder,
                   dataset_hint_dir=FLAGS.train_image_class_hint_folder,
                   dummy_hints=False)
  _convert_dataset('val_class_hints',
                   dataset_dir=FLAGS.val_image_folder,
                   dataset_label_dir=FLAGS.val_image_label_folder,
                   dataset_hint_dir=FLAGS.val_image_class_hint_folder,
                   dummy_hints=False)
  _convert_dataset('test_class_hints',
                   dataset_dir=FLAGS.test_image_folder,
                   dataset_label_dir=FLAGS.test_image_label_folder,
                   dataset_hint_dir=FLAGS.test_image_class_hint_folder,
                   dummy_hints=False)

  _convert_dataset('train_empty_class_hints',
                   dataset_dir=FLAGS.train_image_folder,
                   dataset_label_dir=FLAGS.train_image_label_folder,
                   dataset_hint_dir=FLAGS.train_image_class_hint_folder,
                   dummy_hints=True)
  _convert_dataset('val_empty_class_hints',
                   dataset_dir=FLAGS.val_image_folder,
                   dataset_label_dir=FLAGS.val_image_label_folder,
                   dataset_hint_dir=FLAGS.val_image_class_hint_folder,
                   dummy_hints=True)
  _convert_dataset('test_empty_class_hints',
                   dataset_dir=FLAGS.test_image_folder,
                   dataset_label_dir=FLAGS.test_image_label_folder,
                   dataset_hint_dir=FLAGS.test_image_class_hint_folder,
                   dummy_hints=True)

if __name__ == '__main__':
  tf.app.run()
