# coding=utf-8
# Copyright 2020 Google Health Research.
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
"""Define networks for various EHR data encoders.

Networks are sonnet models that are used in the model construction.
"""
import functools as ft
import itertools
from typing import List, Type

from ehr_prediction_modeling import types
from ehr_prediction_modeling.models.nets import encoder_base
from ehr_prediction_modeling.models.nets import net_utils
import sonnet as snt
import tensorflow.compat.v1 as tf


class FullyConnectedEncoder(encoder_base.DeepEncoderBase):
  """Fully connected embedding that uses initial sparse matrix multiply."""

  def _build(
      self,
      input_data: encoder_base.EncoderInput,
      context: encoder_base.EncoderContext = None
  ) -> tf.Tensor:
    """Evaluate a fully connected encoder.

    Args:
      input_data: a tuple of [indices, values] in this order, both in the
        SparseTensor representation.
      context : optional context or data to be used for conditioning.
      Context is expected to be a dict with fields 'is_training' and
      any other external variables that are needed as side information.

    Returns:
      embedding: a Tensor of embedding of size (batch x ndim_emb).
    """

    embedding = self.initial_embedding(input_data)  # pylint: disable=not-callable
    embedding = self._act_fn(embedding)

    if self._encoder_depth > 0:
      # Add an MLP to the output of the first layer embedding
      layer_sizes = (self._n_encoder_hidden[1:] + [self.embed_out])
      inits = dict(net_utils.INITIALIZER_DICT)


      embedding = snt.nets.MLP(
          layer_sizes,
          initializers=inits,
          activate_final=self._activate_final,
          activation=self._act_fn,
          use_bias=True)(embedding)
    return embedding


class ResidualEncoder(encoder_base.DeepEncoderBase):
  """Residual or Highway network embedding that uses initial sparse multiply."""

  def _build(
      self,
      input_data: encoder_base.EncoderInput,
      context: encoder_base.EncoderContext = None
  ) -> tf.Tensor:
    """Creates a residual encoder.

    This encoder has an initial sparse matrix multiplication, followed by
    several residual layers and then a final linear layer to obtain the final
    embedding size. The depth of the network is n_residual_layers + 2.

    Args:
      input_data: a list of [indices, values] in this order, both in the
        SparseTensor representation.
      context : optional context or data to be used for conditioning.
      This is not used in this network, but must be included to conform to the
      interface used by the generative package. Might be useful later if this
      needs to be conditioned on other inputs.

    Returns:
      embedding: a Tensor of embedding of size (batch x ndim_emb).
    """

    # Initial embedding using sparse matrix multiple
    embedding = self.initial_embedding(input_data)  # pylint: disable=not-callable
    embedding = self._act_fn(embedding)

    # Residual layers if specified in config
    if self._encoder_depth > 0:
      # Residual size will be the same for all, so only need the first entry
      n_residual_size = self._n_encoder_hidden[0]
      residual_fn = ft.partial(
          net_utils.res_block,
          nhidden=n_residual_size,
          highway=self._use_highway_connection)
      res_layer = snt.Module(build=residual_fn, name="res_layer")

      for _ in range(self._encoder_depth):
        embedding = self._act_fn(res_layer(embedding))

      # Add a final layer to get output size of ndim_embed, which gives the
      # mean embedding (and var if needed) and optionally add activation.
      embedding = snt.Linear(output_size=self.embed_out)(embedding)
      if self._activate_final:
        embedding = self._act_fn(embedding)

    return embedding




def get_encoder(encoder_type: str) -> Type[encoder_base.DeepEncoderTypes]:
  """Return encoder function."""
  enc_type_to_fn = {
      types.EmbeddingEncoderType.FC: FullyConnectedEncoder,
      types.EmbeddingEncoderType.RESIDUAL: ResidualEncoder,
  }

  if encoder_type not in enc_type_to_fn:
    raise ValueError(f"Unknown EmbeddingEncoderType: {encoder_type}")
  return enc_type_to_fn[encoder_type]
