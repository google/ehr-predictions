# coding=utf-8
# Copyright 2021 Google Health Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""TensorFlow utility functions to extract Tensor and batch from tf protos."""

import os
from typing import List, Optional
from absl import logging

from ehr_prediction_modeling.data import tf_dataset_utils
from ehr_prediction_modeling.tasks import coordinator
from ehr_prediction_modeling.utils import batches
import tensorflow.compat.v1 as tf

from ehr_prediction_modeling import configdict


class BatchGenerator(object):
  """Tool for working with a TF dataset."""

  def __init__(
      self,
      config: configdict.ConfigDict,
      is_training: bool,
      task_coordinator: Optional[coordinator.Coordinator],
      data_split_name: str,
  ):
    self._config = config
    self._is_training = is_training
    self._task_coordinator = task_coordinator
    self._split_name = data_split_name
    self._iterator = None  # Initialized in create_batch.
    self._batch = self._create_batch()

  @property
  def iterator(self) -> tf.data.Iterator:
    return self._iterator

  @property
  def batch(self) -> batches.TFBatch:
    return self._batch

  def _get_filename(self) -> str:
    split_to_filename = {
        "train": self._config.data.train_filename,
        "valid": self._config.data.valid_filename,
        "test": self._config.data.test_filename,
        "calib": self._config.data.calib_filename,
    }
    return os.path.join(self._config.data.records_dirpath,
                        split_to_filename[self._split_name])

  def _create_batch(self) -> batches.TFBatch:
    """Creates a batch of data in time-major format."""
    tfrecords_filename = self._get_filename()
    if not tfrecords_filename:
      raise ValueError("No recordio found for split {} at {}.".format(
          repr(self._split_name), self._config.recordio_path))
    num_unroll = self._config.data.num_unroll
    batch_size = self._config.data.batch_size
    segment_length = getattr(self._config.data, "segment_length", None)
    if not self._is_training:
      if segment_length is not None:
        logging.info(
            "Splitting sequences by segment_length is only supported during "
            "Training. Updating segment_length to None.")
        segment_length = None

    parallelism_config = self._config.data.padded_settings

    with tf.device("/cpu"):
      raw_dataset = tf_dataset_utils.read_seqex_dataset(tfrecords_filename)
      dataset = transform_dataset(
          raw_dataset,
          task_coordinator=self._task_coordinator,
          parse_cycle_length=parallelism_config.parse_cycle_length,
          context_features=self._config.data.context_features,
          sequential_features=self._config.data.sequential_features,
          batch_size=batch_size,
          num_unroll=num_unroll,
          segment_length=segment_length,
          num_prefetch=parallelism_config.num_prefetch,
          shuffle=self._config.data.shuffle,
      )

      if self._is_training:
        # Don't repeat if we are iterating the whole dataset every eval epoch.
        dataset = dataset.repeat(-1)

      self._iterator = tf.data.make_initializable_iterator(dataset)
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           self._iterator.initializer)
      return self._iterator.get_next()


def transform_dataset(
    dataset: tf.data.Dataset,
    task_coordinator: Optional[coordinator.Coordinator] = None,
    parse_cycle_length: int = 128,
    context_features: Optional[List[str]] = None,
    sequential_features: Optional[List[str]] = None,
    batch_size: int = 32,
    num_unroll: int = 128,
    segment_length: Optional[int] = None,
    num_prefetch: int = 16,
    shuffle: bool = False,
) -> tf.data.Dataset:
  """Transforms the dataset format.

  Args:
    dataset: A Tensorflow Dataset to transform.
    task_coordinator: Coordinator instance with the info about tasks.
    parse_cycle_length: Number of parallel calls to parsing.
    context_features: features to add to the context.
    sequential_features: features to add to the sequence. with lists of indexes
      for each historical feature type.
    batch_size: the batch size.
    num_unroll: the fixed sequence length.
    segment_length: the fixed segments (sub-sequences) length.
    num_prefetch: Number of (batched) sequences to prefetch.
    shuffle: Whether to shuffle the data.

  Returns:
    A TF dataset with context and sequence tensors.
    Each element of this dataset is a Batch object.
  """
  if segment_length is not None and (segment_length % num_unroll) != 0:
    raise ValueError("segment_length should be multiples of num_unroll, "
                     "found segment_length={} and num_unroll={}.".format(
                         segment_length, num_unroll))
  ctx, seq = tf_dataset_utils.get_label_dicts(task_coordinator,
                                              context_features,
                                              sequential_features)

  dataset = dataset.batch(batch_size, drop_remainder=False)
  # Parallelize the parse call
  seqex_to_dict = lambda x: tf_dataset_utils.seqex_to_dict(x, ctx, seq)
  dataset = dataset.map(seqex_to_dict, num_parallel_calls=parse_cycle_length)

  if segment_length is not None:
    dataset = dataset.unbatch()
    dataset = dataset.batch(1, drop_remainder=False)
    # Call convert_to_segments with batch_size=1 to reduce graph size.
    dataset = dataset.map(
        lambda x: tf_dataset_utils.convert_to_segments(x, segment_length),
        num_parallel_calls=parse_cycle_length)
    dataset = dataset.unbatch()
    if shuffle:
      dataset = dataset.shuffle(buffer_size=batch_size * 64)
    dataset = dataset.batch(batch_size, drop_remainder=False)

  dataset = dataset.map(
      tf_dataset_utils.convert_to_time_major,
      num_parallel_calls=parse_cycle_length)
  absmp = tf_dataset_utils.add_beginning_of_sequence_mask_and_pad
  add_beginning_of_sequence_mask_and_pad = (
      lambda x: absmp(x, batch_size, num_unroll))
  dataset = dataset.flat_map(add_beginning_of_sequence_mask_and_pad)
  # Rebatch the data by num_unroll
  dataset = dataset.unbatch()
  dataset = dataset.batch(num_unroll, drop_remainder=False)

  if num_prefetch != -1:
    dataset = dataset.prefetch(num_prefetch)

  return dataset
