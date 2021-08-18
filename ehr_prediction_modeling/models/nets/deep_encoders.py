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
"""Define networks for various EHR data encoders.

Networks are sonnet models that are used in the model construction.
"""
import functools as ft
import itertools
from typing import List, Type

from ehr_prediction_modeling import types
from ehr_prediction_modeling.models.nets import encoder_base
from ehr_prediction_modeling.models.nets import net_utils
from ehr_prediction_modeling.models.nets import snr
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

      act_fn = ft.partial(
          net_utils.activation_with_batch_norm,
          act_fn=self._act_fn,
          context=context,
          use_bn=self._use_batch_norm)

      # If using batch_norm, no need for a bias, since it remove the effect.
      if self._use_batch_norm: del inits["b"]

      embedding = snt.nets.MLP(
          layer_sizes,
          initializers=inits,
          activate_final=self._activate_final,
          activation=act_fn,
          use_bias=not self._use_batch_norm,
          use_dropout=(self._dropout_prob > 0))(
              embedding,
              is_training=self._dropout_is_training,
              dropout_keep_prob=(1 - self._dropout_prob))
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
      act_fn = ft.partial(
          net_utils.activation_with_batch_norm,
          act_fn=self._act_fn,
          context=context,
          use_bn=self._use_batch_norm)
      # Residual size will be the same for all, so only need the first entry
      n_residual_size = self._n_encoder_hidden[0]
      residual_fn = ft.partial(
          net_utils.res_block,
          nhidden=n_residual_size,
          highway=self._use_highway_connection,
          use_bias=not self._use_batch_norm,
          dropout_prob=self._dropout_prob,
          is_training=self._dropout_is_training)
      res_layer = snt.Module(build=residual_fn, name="res_layer")

      for _ in range(self._encoder_depth):
        embedding = act_fn(res_layer(embedding))

      # Add a final layer to get output size of ndim_embed, which gives the
      # mean embedding (and var if needed) and optionally add activation.
      embedding = snt.Linear(output_size=self.embed_out)(embedding)
      if self._activate_final:
        embedding = self._act_fn(embedding)

    return embedding


class SNREncoder(encoder_base.DeepEncoderBase):
  """Sub-network routing encoder using i=initial sparse matrix multiply."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._snr_encoder_layers = []

  @property
  def embed_out(self) -> int:
    """Returns the output embedding size for pres and num features."""
    return self._ndim_emb

  @property
  def initial_embed_size(self) -> int:
    if self._encoder_depth > 0:
      embed_size = self._n_encoder_hidden[0][0]
    else:
      embed_size = self.embed_out

    if isinstance(embed_size, list):
      if len(embed_size) != 1:
        raise ValueError(f"ndim_emb is a list which means that the "
                         f"SparseLookupEncoder is applied before the "
                         f"SNREncoder. The length of the list should be equal "
                         f"to 1. Found length {len(embed_size)}")
      initial_embed_size = embed_size[-1]
    else:
      initial_embed_size = embed_size
    return initial_embed_size

  @property
  def _tasks(self) -> List[str]:
    return sorted(self._emb_config.get("tasks", []))

  def get_task_routing_connections(self):
    """Returns a dictionary of task specific routing connections.

    If use_task_specific_routing is set to false then the dictionary will be
    empty as there are no task specific connections created.
    """
    if not self._tasks:
      return {}

    encoder_variables = [
        w for w in tf.trainable_variables() if "enc" in w.name.lower()
    ]
    task_routing_connections = {}
    for task_name in self._tasks:
      snr_connections = [
          w for w in encoder_variables
          if f"task_{task_name.lower()}" in w.name.lower() and
          "routing" in w.name.lower()
      ]
      task_routing_connections[task_name] = snr_connections
    return task_routing_connections

  def get_regularization_penalty(self) -> tf.Tensor:
    """Compute embedding regularization penalty for SNREncoder.

    SNR Encoder is composed of three kinds of weight variables - weights of the
    SparseLookUpEncoder, weights of the sub-networks, and the connections
    between different sub-networks. This section separates these variables and
    applies regularisation.

    Returns:
      Embedding regularization penalty.
    """
    return sum([
        (snr_layer.get_snr_connections_loss() + snr_layer.get_weights_penalty())
        for snr_layer in self._snr_encoder_layers
    ])

  def _build(
      self,
      input_data: encoder_base.EncoderInput,
      context: encoder_base.EncoderContext = None
  ) -> tf.Tensor:
    """Evaluate a SNR encoder.

    Args:
      input_data: a list of [indices, values] in this order, both in the
        SparseTensor representation.
      context : optional context or data to be used for conditioning. Context is
        expected to be a dict with fields 'is_training' and any other external
        variables that are needed as side information.

    Returns:
      embedding: a Tensor of embedding of size (batch x ndim_emb).
    """

    embedding = self.initial_embedding(input_data)  # pylint: disable=not-callable
    # No batch norm on initial embedding.

    inits = dict(net_utils.INITIALIZER_DICT)
    batch_size = tf.shape(embedding)[0]
    with tf.variable_scope(self._emb_config.encoder_type):
      embedding = [embedding]

      act_fn = ft.partial(
          net_utils.activation_with_batch_norm,
          act_fn=self._act_fn,
          context=context,
          use_bn=self._use_batch_norm)

      # Define the SNR forward pass.
      networks = self._n_encoder_hidden[1:]
      # For each layer.
      for i in range(len(networks)):
        with tf.variable_scope("layer_%s" % i):
          layer_activations = []
          # For each sub-network.
          for j in range(len(networks[i])):
            with tf.variable_scope("subnetwork_%s" % j):
              snr_encoder_layer = snr.SNREncoderLayer(
                  subnetwork_coordinates=(i, j),
                  nhidden=networks[i][j],
                  batch_size=batch_size,
                  activation_function=act_fn,
                  snr_config=self._emb_config.snr,
                  tasks=self._tasks,
                  initializers=inits,
                  dropout_prob=self._dropout_prob,
                  use_bias=not self._use_batch_norm,
                  is_training=self._dropout_is_training,
                  is_final_layer=(i == (len(networks) - 1)),
                  activate_final=self._activate_final)
              self._snr_encoder_layers.append(snr_encoder_layer)
              subnetwork_activations = snr_encoder_layer(embedding)
              layer_activations.append(subnetwork_activations)
          # The following snippet optionally adds Boolean skip connections
          # across the layers of the SNR Encoder.
          if self._emb_config.snr.use_skip_connections and i != 0:
            embedding.extend(layer_activations)
          else:
            embedding = layer_activations
      return embedding


def get_snr_embed_out_size(config, feat):
  """Compute the size of the embedding layer output for SNREncoder."""

  def _snr_embed_out_skip(feat_layer_size):
    if config.deep.snr.use_skip_connections:
      return list(itertools.chain(*feat_layer_size[1:]))
    else:
      return feat_layer_size[-1]

  size_multiplier = 1
  if (config.deep.snr.use_task_specific_routing and
      config.deep.snr.input_combination_method ==
      types.SNRInputCombinationType.CONCATENATE):
    size_multiplier = len(config.deep.tasks)

  pres_size, num_size, dom_size = config.deep.encoder_layer_sizes
  if feat in [
      types.FeatureTypes.PRESENCE_HIST,
      types.FeatureTypes.PRESENCE_RECENT,
      types.FeatureTypes.PRESENCE_SEQ]:
    return size_multiplier * _snr_embed_out_skip(pres_size)
  elif feat in [
      types.FeatureTypes.NUMERIC_HIST,
      types.FeatureTypes.NUMERIC_RECENT,
      types.FeatureTypes.NUMERIC_SEQ]:
    return size_multiplier * _snr_embed_out_skip(num_size)
  elif feat in [
      types.FeatureTypes.DOMAIN_HIST,
      types.FeatureTypes.DOMAIN_RECENT,
      types.FeatureTypes.DOMAIN_SEQ]:
    return size_multiplier * _snr_embed_out_skip(dom_size)
  else:
    raise ValueError("Wrong feature type encountered. Should be one of the"
                     "feature types defined in types.FeatureTypes")


def get_snr_embedding_dim_dict(config):
  """Get the feature-wise dimension of each embedding module."""
  embedding_dim_dict = {}
  for feat in config.identity_lookup_features:
    embedding_dim_dict[feat] = [config.ndim_dict[feat]]
  for feat in config.context_features + config.sequential_features:
    if feat not in embedding_dim_dict:
      embedding_dim_dict[feat] = get_snr_embed_out_size(config, feat)
  return embedding_dim_dict


def compute_combined_snr_embedding(embedding_dict):
  combined_embedding = []
  for feat_type_embed_out in embedding_dict.values():
    combined_embedding.extend(
        feat_type_embed_out if isinstance(
            feat_type_embed_out, list) else [feat_type_embed_out])
  return combined_embedding


def get_encoder(encoder_type: str) -> Type[encoder_base.DeepEncoderTypes]:
  """Return encoder function."""
  enc_type_to_fn = {
      types.EmbeddingEncoderType.FC: FullyConnectedEncoder,
      types.EmbeddingEncoderType.RESIDUAL: ResidualEncoder,
      types.EmbeddingEncoderType.SNR: SNREncoder,
  }

  if encoder_type not in enc_type_to_fn:
    raise ValueError(f"Unknown EmbeddingEncoderType: {encoder_type}")
  return enc_type_to_fn[encoder_type]
