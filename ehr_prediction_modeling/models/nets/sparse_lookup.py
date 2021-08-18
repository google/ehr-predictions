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
"""Sparse Lookup Encoder, for use as a basic lookup embedding or by other encoders."""
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


VALID_COMBINERS = ["sum", "sqrtn", "mean"]


def combiner_correction(sparse_combiner: str, embed: tf.Tensor) -> tf.Tensor:
  """Corrections for fixing NaNs in certain combining methods.

  There are many zero-only rows, which result in NaN/inf entries when using a
  sparse_combine method other than "sum", because it divides by zero.
  Here, this checks for NaNs and replaces them with zeros.
  Must be careful here that any NaNs that enter before this point are not
  hidden.

  Args:
    sparse_combiner: The embedding reduction method.
    embed: The embedding.

  Returns:
    The embedding without NaN values (they are replaced with zeros).
  """
  if sparse_combiner in VALID_COMBINERS:
    embed = tf.where(tf.is_nan(embed), tf.zeros_like(embed), embed)
  return embed


class SparseLookupEncoder(snt.AbstractModule):
  """A lookup embedding based on lookup table to be used by other classes."""

  def __init__(self,
               ndim_emb: int,
               ndim_input: int,
               n_act: Optional[int] = None,
               activate_final: bool = False,
               act_fn: Callable[[tf.Tensor], tf.Tensor] = tf.nn.tanh,
               sparse_lookup_dropout_prob: float = 0.,
               dropout_is_training: bool = True,
               sparse_combine: str = "sum",
               name: Optional[str] = None,
               identity_lookup: bool = False):

    if name is None:
      name = "SparseLookupEncoder"

    super().__init__(name=name)
    self._ndim_emb = ndim_emb
    self._ndim_input = ndim_input
    self._n_act = 0 if not n_act else n_act
    self._act_fn = act_fn
    self._activate_final = activate_final
    self._dropout_prob = sparse_lookup_dropout_prob
    self._dropout_is_training = dropout_is_training

    # Apply a correction by diving by the square root or mean of the number of
    # observed entries to control variance scaling the number of obs entries.
    self._sparse_combine = sparse_combine
    if identity_lookup:
      # Just use a identity matrix lookup table
      if (self._ndim_emb % self._ndim_input) != 0:
        raise ValueError(
            "SparseEmbeddingLookup error: Embedding size for identity lookup "
            "feature should be an integer factor of input size.")
      # When using a distributional output, the number of outputs must be
      # expanded to model each of the distribution variables. E.g.,
      # for Gaussians, the number of distributions parameters will be 2
      # for the mean and std_dev. This is needed specifically for the domain
      # embeddings since their output dimension is not changed.
      num_output_parameters = int(self._ndim_emb/self._ndim_input)

      with self._enter_variable_scope():
        weights = num_output_parameters*[tf.eye(self._ndim_input)]
        self._w = tf.concat(weights, axis=1)

    else:
      # Initialise randomly for presence and numeric lookup table.
      init_val = np.sqrt(6. / (self._n_act + self._ndim_emb))
      weight_shape = [self._ndim_input, self._ndim_emb]
      initializer = tf.random_uniform_initializer(minval=-init_val,
                                                  maxval=init_val)
      with self._enter_variable_scope():
        self._w = tf.get_variable(
            "init_embed", shape=weight_shape, initializer=initializer)
      # Add dropout
      if self._dropout_prob > 0.:
        self._w = slim.dropout(
            self._w,
            keep_prob=(1 - self._dropout_prob),
            noise_shape=[self._ndim_input, 1],
            is_training=self._dropout_is_training)

  def _build(
      self,
      input_data: Tuple[tf.SparseTensor, tf.SparseTensor],
      context: Optional[Dict[str, Union[tf.Tensor, bool,
                                        tf.SparseTensor]]] = None
  ) -> tf.Tensor:
    """Embedding based on a single sparse matmul.

    Args:
      input_data: A tuple of [indices, values] in this order, both in the
        SparseTensor representation.
      context : Optional context or data to be used for conditioning.
      This is not used in this network, but must be included to conform to the
      standard interface.

    Returns:
      The embedding, a Tensor of size (batch x ndim_emb).
    """
    del context  # unused in this network.
    # Unpack input data
    indices, values = input_data

    # Initial sparse matrix multiplication, with sparse gradients
    max_index = tf.reduce_max(tf.to_int32(indices.values))
    num_index = tf.shape(self._w)[0]
    check_index = tf.Assert(
        tf.less(max_index, num_index), [max_index, num_index],
        name="check_index_into_embedding")
    with tf.control_dependencies([check_index]):
      embedding = tf.nn.embedding_lookup_sparse(
          self._w, tf.cast(indices, tf.int64), values,
          combiner=self._sparse_combine)

    embedding = combiner_correction(self._sparse_combine, embedding)

    # Some data augmentations are implemented through pyfuncs and make
    # the shape unknown here. We set the shape manually so that later stages
    # of network can create the correctly shaped embedding.
    embedding.set_shape([None, self._w.get_shape().as_list()[1]])

    if self._activate_final:
      embedding = self._act_fn(embedding)

    return embedding
