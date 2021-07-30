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
from typing import Dict, List, Optional, Tuple, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.tasks import coordinator
from ehr_prediction_modeling.utils import batches
from ehr_prediction_modeling.utils import label_utils
import tensorflow.compat.v1 as tf


SeqexDictType = Dict[str, Union[tf.Tensor, Dict[str, Union[tf.Tensor,
                                                           tf.SparseTensor]]]]


def sample_segments(
    seq_lens: tf.Tensor,
    segment_length: tf.Tensor,
    random_offset: bool,
) -> tf.Tensor:
  """Sample fixed length segments from a list of sequence lengths.

  Args:
    seq_lens: A 1D tensor of sequence lengths.
    segment_length: Length of sampled segments.
    random_offset: Whether to apply a per-sequence random offset to the start of
      each segments.

  Returns:
    A 2D tensor where each row contains two elements: the sequence index and the
    starting time/index of a segment.
  """
  seq_lens = tf.cast(seq_lens, tf.int32)
  num_elems = tf.shape(seq_lens)[0]

  max_seq_len = tf.reduce_max(seq_lens + segment_length)
  steps = tf.range(max_seq_len, delta=segment_length)

  if random_offset:
    # Apply a per-sequence offset.
    tiled_steps = tf.expand_dims(steps, axis=0)
    tiled_steps -= tf.random.uniform((num_elems, 1),
                                     maxval=segment_length,
                                     dtype=tf.int32)
    tiled_steps = tf.maximum(0, tiled_steps)
  else:
    tiled_steps = tf.tile([steps], [num_elems, 1])

  mask = tiled_steps <= tf.expand_dims(seq_lens, axis=-1)
  mask.shape.assert_has_rank(2)

  def flatten(x):
    return tf.reshape(x, [-1])

  num_steps = tf.shape(steps)[0]
  tiled_indices = tf.tile(
      tf.expand_dims(tf.range(num_elems), axis=-1), [1, num_steps])
  segment_starts = tf.stack(
      [flatten(tiled_indices), flatten(tiled_steps)], axis=-1)
  segment_starts = segment_starts[flatten(mask)]

  return segment_starts


def slice_tensor(
    tensor: tf.Tensor,
    seq_starts: tf.Tensor,
    segment_length: int,
) -> tf.Tensor:
  """Slice segments from a tensor [batch x num_unrol x feats]."""
  num_elems = tf.shape(seq_starts)[0]
  init_array = tf.TensorArray(tensor.dtype, size=num_elems)

  # pad upto multiples of segment length
  batch_size = tf.shape(tensor)[0]
  padded_shape = tf.concat(
      [[tf.cast(batch_size, tf.int64)], [tf.cast(segment_length, tf.int64)],
       tf.shape(tensor)[2:]],
      axis=0)
  padded_tensor = tf.concat(
      [tensor, tf.zeros(shape=padded_shape, dtype=tensor.dtype)], axis=1)

  # writes each sliced tensor into TensorArray ta at position i
  def loop_body(i, ta):
    batch_idx = seq_starts[i, 0]
    start_idx = seq_starts[i, 1]
    indices = tf.expand_dims(
        tf.range(start_idx, start_idx + segment_length), axis=-1)
    vals = tf.gather_nd(padded_tensor[batch_idx, ...], indices)
    return i + 1, ta.write(i, vals)

  _, result_array = tf.while_loop(lambda i, ta: i < num_elems, loop_body,
                                  [0, init_array])

  new_tensor = result_array.stack(name="stack_slice_tensor")

  return new_tensor


def slice_sparse_tensor_sequences(
    tensor: tf.SparseTensor,
    seq_starts: tf.Tensor,
    segment_length: int,
) -> tf.SparseTensor:
  """Slice segments from a sparse tensor [batch x num_unrol x feats]."""
  seq_starts = tf.cast(seq_starts, tf.int64)
  num_segments = tf.cast(tf.shape(seq_starts)[0], tf.int64)
  num_sparse_rows = tf.cast(tf.shape(tensor.indices)[0], tf.int64)

  expand_before = lambda x: tf.expand_dims(x, axis=0)
  expand_after = lambda x: tf.expand_dims(x, axis=-1)

  # Mask is [num_elems x len(tensor.values)], indicating for each new batch,
  # which (row of) values in the sparse tensor is included.
  # This tensor can be become very very large for large batch size.
  # Match batch indices.
  mask = tf.equal(
      expand_before(tensor.indices[:, 0]), expand_after(seq_starts[:, 0]))
  # Match segment start times.
  mask &= (
      expand_before(tensor.indices[:, 1]) >= expand_after(seq_starts[:, 1]))
  # Match segment end times.
  mask &= (
      expand_before(tensor.indices[:, 1]) <
      expand_after(seq_starts[:, 1] + segment_length))

  # Each row of old_time_indices is tf.range(num_sparse_rows)
  _, old_time_idx = tf.meshgrid(
      tf.range(num_segments), tf.range(num_sparse_rows), indexing="ij")
  # Each column of segment_idx is tf.range(num_segments)
  # Each row of new_time_idx is the time column of tensor.indices
  segment_idx, new_time_idx = tf.meshgrid(
      tf.range(num_segments), tensor.indices[:, 1], indexing="ij")
  # Offset the start of each segment to 0
  new_time_idx -= expand_after(seq_starts[:, 1])

  flat_mask = tf.reshape(mask, [-1])
  flat_segment_idx = tf.reshape(segment_idx, [-1, 1])[flat_mask]
  flat_new_time_idx = tf.reshape(new_time_idx, [-1, 1])[flat_mask]
  flat_old_time_idx = tf.reshape(old_time_idx, [-1, 1])[flat_mask]

  # Construct new sparse tensor.
  new_indices = tf.gather_nd(tensor.indices, flat_old_time_idx)
  new_indices = tf.concat(
      [flat_segment_idx, flat_new_time_idx, new_indices[:, 2:]], axis=-1)
  new_values = tf.gather_nd(tensor.values, flat_old_time_idx)
  new_shape = (num_segments, segment_length, tensor.dense_shape[-1])

  new_tensor = tf.SparseTensor(new_indices, new_values, new_shape)

  return new_tensor


def slice_sparse_tensor_context(tensor: tf.SparseTensor,
                                seq_starts: tf.Tensor) -> tf.SparseTensor:
  """Slice segments from a sparse tensor [batch x feats]."""
  seq_starts = tf.cast(seq_starts, tf.int64)
  num_segments = tf.cast(tf.shape(seq_starts)[0], tf.int64)
  num_sparse_rows = tf.cast(tf.shape(tensor.indices)[0], tf.int64)

  expand_before = lambda x: tf.expand_dims(x, axis=0)
  expand_after = lambda x: tf.expand_dims(x, axis=-1)

  # Mask is [num_elems x len(tensor.values)], indicating for each new batch,
  # which (row of) values in the sparse tensor is included.
  # This tensor can become very very large for large batch size.
  mask = tf.equal(
      expand_before(tensor.indices[:, 0]), expand_after(seq_starts[:, 0]))

  # Each column of segment_idx is tf.range(num_segments)
  segment_idx, row_idx = tf.meshgrid(
      tf.range(num_segments), tf.range(num_sparse_rows), indexing="ij")

  flat_mask = tf.reshape(mask, [-1])
  flat_segment_idx = tf.reshape(segment_idx, [-1, 1])[flat_mask]
  flat_row_idx = tf.reshape(row_idx, [-1, 1])[flat_mask]

  # Construct new sparse tensor.
  new_indices = tf.gather_nd(tensor.indices, flat_row_idx)
  new_indices = tf.concat([flat_segment_idx, new_indices[:, 1:]], axis=-1)
  new_values = tf.gather_nd(tensor.values, flat_row_idx)
  new_shape = (num_segments, tensor.dense_shape[-1])

  new_tensor = tf.SparseTensor(new_indices, new_values, new_shape)

  return new_tensor


def convert_to_segments(
    examples: SeqexDictType,
    segment_len: int,
) -> SeqexDictType:
  """Parses the serialized example into time-major sequence and context."""

  starts = sample_segments(examples["sequence_lengths"], segment_len, False)

  res = {"context": {}, "sequences": {}}
  for key, tensor in examples["sequences"].items():
    if isinstance(tensor, tf.Tensor):
      res["sequences"][key] = slice_tensor(tensor, starts, segment_len)
    elif isinstance(tensor, tf.SparseTensor):
      res["sequences"][key] = slice_sparse_tensor_sequences(
          tensor, starts, segment_len)
    else:
      raise ValueError(
          "Unexpected element in batch.sequences {}".format(tensor))

  for key, tensor in examples["context"].items():
    if isinstance(tensor, tf.Tensor):
      res["context"][key] = tf.gather_nd(tensor, starts[:, :1])
    elif isinstance(tensor, tf.SparseTensor):
      res["context"][key] = slice_sparse_tensor_context(tensor, starts)
    else:
      raise ValueError("Unexpected element in batch.context {}".format(tensor))

  return res


def read_seqex_dataset(tfrecords_filename: str) -> tf.data.TFRecordDataset:
  return tf.data.TFRecordDataset([tfrecords_filename])


def tile_context(
    ctx: Dict[str, tf.Tensor],
    seq_len: int,
    batch_padding: Optional[int] = None,
) -> Dict[str, tf.Tensor]:
  """Tiles the context along the time dimension seq_len times and pads batch."""
  # Note: Currently only supports dense tensors with feature dimension = 1.
  tiled_ctx = {}
  for key, tensor in ctx.items():
    ctx_multiplier = tf.concat(
        [[seq_len], tf.ones([tf.rank(tensor) - 1], dtype=tf.int32)], 0)
    # Make context tensor the same shape as the sequence tensor by tiling
    # by the time dimension
    ctx_reshaped = tf.reshape(
        tf.tile(tensor, ctx_multiplier),
        tf.concat([[seq_len], tf.shape(tensor)], 0))
    if batch_padding is not None:
      # Pad the final batch in the dataset
      ctx_reshaped = tf.concat([
          ctx_reshaped,
          tf.zeros([seq_len, batch_padding], dtype=ctx_reshaped.dtype)
      ], 1)
    tiled_ctx[key] = ctx_reshaped
  return tiled_ctx


def add_beginning_of_sequence_mask_and_pad(
    example: SeqexDictType,
    batch_size: int,
    num_unroll: int,
) -> tf.data.Dataset:
  """Pad sequence to be divisible by num_unroll and set end of sequence mask."""
  ctx = example["context"]
  seq = example["sequences"]
  seq_len = tf.shape(seq[label_utils.TIMESTAMP_KEY])[0]
  batch_len = tf.shape(seq[label_utils.TIMESTAMP_KEY])[1]
  seq_len_padding = tf.cond(
      tf.equal(seq_len % num_unroll, 0), lambda: 0,
      lambda: num_unroll - seq_len % num_unroll)
  # Pad the final batch in the dataset
  batch_padding = tf.cond(
      tf.equal(batch_len, batch_size), lambda: 0,
      lambda: batch_size - batch_len)
  padded_seq_len = seq_len + tf.cast(seq_len_padding, tf.int32)
  for key, tensor in seq.items():
    if isinstance(tensor, tf.Tensor):
      # Pad sequence to be of length divisible by num_unroll
      padded_shape = tf.concat(
          [[seq_len_padding], tf.shape(tensor)[1:]], axis=0)
      padded_tensor = tf.concat(
          [tensor, tf.zeros(shape=padded_shape, dtype=tensor.dtype)], 0)
      # Pad final sequence to full batch size
      padded_tensor = tf.concat([
          padded_tensor,
          tf.zeros([padded_seq_len, batch_padding,
                    tf.shape(padded_tensor)[2]],
                   dtype=padded_tensor.dtype)
      ], 1)
      seq[key] = padded_tensor
    elif isinstance(tensor, tf.SparseTensor):
      # Update time dimension of dense shape to padded length
      padded_shape = tf.concat(
          [[tf.cast(padded_seq_len, tf.int64)], [tf.cast(batch_size, tf.int64)],
           tensor.dense_shape[2:]], 0)
      # Reinstate canonical ordering of sparse tensor.
      seq[key] = tf.sparse_reorder(
          tf.SparseTensor(
              indices=tensor.indices,
              values=tensor.values,
              dense_shape=padded_shape))
    else:
      continue

  ctx = tile_context(ctx, padded_seq_len, batch_padding)
  beginning_of_sequence_mask = tf.concat([
      tf.fill([1, batch_len + batch_padding], True),
      tf.fill([seq_len + seq_len_padding - 1, batch_len + batch_padding], False)
  ], 0)

  return tf.data.Dataset.from_tensors(
      batches.TFBatch(
          context=ctx,
          sequences=seq,
          is_beginning_sequence=beginning_of_sequence_mask))


def seqex_to_dict(
    serialized_examples: bytes,
    context_d: Dict[str, str],
    sequence_d: Dict[str, str],
) -> SeqexDictType:
  """Parses the serialized example into time-major sequence and context."""
  ctx, seq, seq_lens = tf.io.parse_sequence_example(
      serialized_examples,
      context_features=context_d,
      sequence_features=sequence_d)

  res = {
      "context": ctx,
      "sequences": seq,
      "sequence_lengths": seq_lens[label_utils.TIMESTAMP_KEY],
  }
  return res


def get_label_dicts(
    task_coordinator: coordinator.Coordinator,
    context_features: List[str],
    sequential_features: List[str],
) -> Tuple[Dict[str, Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]],
           Dict[str, Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]]]:
  """Gets the feature dictionaries to parse a tf.SequenceExample.

  Args:
    task_coordinator: tasks.Coordinator instance with the info about tasks.
    context_features: list of features to extract from the context.
    sequential_features: list of features to extract from the sequence.

  Returns:
    context_d: Dictionaries of TF features to read in context.
    sequence_d: Dictionaries of TF features to read in sequence.
  """
  # Dictionary of context.
  context_d = {
      "record_number": tf.FixedLenFeature([], tf.string),
  }
  for feat in context_features:
    for prefix in ["indexes", "values"]:
      tfready_name = "_".join([prefix, types.FEAT_TO_NAME[feat]])
      context_d.update({tfready_name: tf.VarLenFeature(tf.float32)})

  # Dictionary of sequences.
  sequence_d = {
      "ignore_label": tf.FixedLenSequenceFeature([1], tf.int64),
      "delta_time": tf.FixedLenSequenceFeature([1], tf.int64),
      label_utils.TIMESTAMP_KEY: tf.FixedLenSequenceFeature([1], tf.int64),
      "episode_id": tf.FixedLenSequenceFeature([1], tf.string)
  }
  for feat in sequential_features:
    for prefix in ["indexes", "values"]:
      tfready_name = "_".join([prefix, types.FEAT_TO_NAME[feat]])
      sequence_d.update({tfready_name: tf.VarLenFeature(tf.float32)})

  if task_coordinator:
    tasks_context_d, tasks_sequence_d = task_coordinator.get_label_dicts()
    context_d.update(tasks_context_d)
    sequence_d.update(tasks_sequence_d)
  return context_d, sequence_d


def to_time_major(tensor: tf.Tensor) -> tf.Tensor:
  """Transposes batch majored tensors to time major."""
  dimensions = tf.nest.map_structure(lambda x: x.get_shape().ndims, tensor)

  def _transpose(value, dimension):
    if dimension > 1:
      # Swap the first dimension (batch) with the second (time)
      return tf.transpose(value, [1, 0] + list(range(2, dimension)))
    return value

  return tf.nest.map_structure(_transpose, tensor, dimensions)


def sparse_tensor_to_time_major(
    sparse_tensor: tf.SparseTensor) -> tf.SparseTensor:
  """Transposes batch major sparse tensors to time major."""

  time_major_indices = tf.gather(sparse_tensor.indices, [1, 0, 2], axis=1)
  time_major_dense_shape = [
      sparse_tensor.dense_shape[1], sparse_tensor.dense_shape[0],
      sparse_tensor.dense_shape[2]
  ]
  return tf.SparseTensor(time_major_indices, sparse_tensor.values,
                         time_major_dense_shape)


def convert_to_time_major(examples: SeqexDictType) -> SeqexDictType:
  """Parses the serialized example into time-major sequence and context."""

  res = {"context": {}, "sequences": {}}
  for key, tensor in examples["sequences"].items():
    if isinstance(tensor, tf.Tensor):
      res["sequences"][key] = to_time_major(tensor)
    elif isinstance(tensor, tf.SparseTensor):
      res["sequences"][key] = sparse_tensor_to_time_major(tensor)
    else:
      raise ValueError(
          "Unexpected element in batch.sequences {}".format(tensor))

  for key, tensor in examples["context"].items():
    if isinstance(tensor, (tf.Tensor, tf.SparseTensor)):
      res["context"][key] = tf.identity(tensor)
    else:
      raise ValueError("Unexpected element in batch.context {}".format(tensor))

  return res
