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

import collections
from typing import Callable, Dict, List, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.utils import label_utils
import numpy as np
import tensorflow.compat.v1 as tf

# sequences: These are time-varying information. Dict[str, tf.SparseTensor OR
#   tf.Tensor]. This typically maps labels, features, and masks to
#   tf.SparseTensor or tf.Tensor
# context: These are constant information (like gender or ethnicity) Dict[str,
#   tf.Tensor].
# is_beginning_sequence: Boolean values that indicates which timesteps are the
#   beginning of a patient's medical history. It is used so that the RNN
#   hidden state can be reset between patients.
TFBatch = collections.namedtuple(
    "Batch", ["sequences", "context", "is_beginning_sequence"])



def batch_to_components(
    batch: TFBatch,
    context_features: List[str],
    sequential_features: List[str],
) -> Dict[str, tf.Tensor]:
  """Unpack data into constituent components.

  Args:
    batch: iterator to provide data.
    context_features: list of features required from the context. e.g. ["pres",
      "num", "dom"]
    sequential_features: list of features required from the sequence. e.g.
      ["pres_s", "num_s", "dom_s"]

  Returns:
    Dictionary of str (feature types) to tf.SparseTensor of dimension
    (num_unroll, batch_size, d).

    ----------------------------------------------------------------------------

    For each feature_type (t, dt), there is a tf.Tensor associated storing all
    the information.

    ----------------------------------------------------------------------------

    For each feature_type <feat_type> (dom, dom_s, dom_recent, num,
    num_s, num_recent, pres, pres_s, pres_recent), there are two tf.SparseTensor
    associated in the dict: idx_<feat_type> and val_<feat_type> (called idx and
    val in what follows for clarity). The information stored in idx and val is
    used to represent some very sparse matrix M in a compact format.

    idx and val are of type tf.SparseTensor with 3 attributes `dense_shape`,
    `values`, and `indices`.

    The following holds:
      1) The dimensions are the same:
           np.all(val.dense_shape.numpy() == idx.dense_shape.numpy())

      2) The indices are the same:
           np.all(val.indices.numpy() == idx.indices.numpy())

      3) If (M_{t,b,d})_{t<T, b<batch_size, d<n_features} is the matrix
         represented by idx and val, and that it has K nonzero values named
         x_0, ..., x_K-1, we have that:
           for all k < K:
             M_{t_k, b_k, d_k} = x_k,
           with:
             t_k == idx.indices[k][0] == val.indices[k][0]
             b_k == idx.indices[k][1] == val.indices[k][1]
             d_k == idx.values[k]
             x_k == val.values[k]
  """
  with tf.device("/cpu"):
    outputs = _batch_to_components(batch, context_features, sequential_features)
  outputs["dt"] = batch.sequences["delta_time"]
  outputs["t"] = batch.sequences[label_utils.TIMESTAMP_KEY]

  return outputs


def _batch_to_components(
    batch: TFBatch,
    context_features: List[str],
    sequential_features: List[str],
) -> Dict[str, tf.Tensor]:
  """Unpack data into constituent components.

  Args:
    batch: iterator to provide data.
    context_features: list of features required from the context. e.g. ["pres",
      "num", "dom"]
    sequential_features: list of features required from the sequence. e.g.
      ["pres_s", "num_s", "dom_s"]

  Returns:
    Dictionary of str (feature types) to tf.SparseTensor of dimension
    (num_unroll, batch_size, d).
  """
  outputs = {}
  input_to_output_prefix = {"values": "val", "indexes": "idx"}
  for inp_prefix, outp_prefix in input_to_output_prefix.items():
    outputs.update(
        _extract_features(batch.context, context_features, outp_prefix,
                          inp_prefix))
    outputs.update(
        _extract_features(batch.sequences, sequential_features, outp_prefix,
                          inp_prefix))
  return outputs


def _extract_features(
    batch_field: Dict[str, tf.Tensor],
    features: List[str],
    outp_prefix: str,
    inp_prefix: str,
) -> Dict[str, tf.Tensor]:
  """Extracts features from `context` or `sequences` field of the seqex."""
  outputs = {}
  for feat in features:
    feat_name = types.FEAT_TO_NAME[feat]
    outputs[outp_prefix + "_" + feat] = (
        batch_field[inp_prefix + "_" + feat_name])
  return outputs


def flatten_batch(val: tf.Tensor) -> tf.Tensor:
  """Flatten a ND Tensor into a 2D Tensor."""
  if isinstance(val, tf.SparseTensor):
    val_shape = val.dense_shape
    flat_val = tf.sparse_reshape(val, [-1, val_shape[-1]])
  else:
    val_shape = tf.shape(val)
    flat_val = tf.reshape(val, [-1, val_shape[-1]])

  return flat_val


def get_unflatten_batch_fn(
    batch_shape: tf.Tensor) -> Callable[[tf.Tensor], tf.Tensor]:
  """Returns a function that reshape its input back into batch_shape."""

  def unflatten_batch(val):
    new_shape = tf.concat([batch_shape, val.shape.as_list()[-1:]], axis=0)
    if isinstance(val, tf.SparseTensor):
      return tf.sparse_reshape(val, new_shape)
    else:
      return tf.reshape(val, new_shape)

  return unflatten_batch


def sparse_fill_empty_rows(
    val: Union[tf.Tensor,
               tf.SparseTensor]) -> Union[tf.Tensor, tf.SparseTensor]:
  """If input is a sparse tensor, fill empty rows with 0."""
  if isinstance(val, tf.SparseTensor):
    return tf.sparse_fill_empty_rows(val, 0)[0]
  else:
    return val
