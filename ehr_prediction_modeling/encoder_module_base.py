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
"""Encoder for TFReady data representation."""
import abc
import typing
from typing import Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union

from ehr_prediction_modeling.models import model_utils
from ehr_prediction_modeling.utils import batches
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim
import tree as nest

if typing.TYPE_CHECKING:
  from ehr_prediction_modeling.embeddings import input_embedding_base
  from ehr_prediction_modeling import configdict

EmbedDimDict = Dict[str, Union[List[int], int]]


class EncoderModule(metaclass=abc.ABCMeta):
  """Encoder for TFReady data representation."""

  def __init__(self, config: "configdict.ConfigDict",
               embedding_classes: Mapping[
                   str, Callable[["configdict.ConfigDict", EmbedDimDict],
                                 "input_embedding_base.InputEmbeddingBase"]]):
    self._config = config
    self._embedding_classes = embedding_classes
    self._embed_dim_dict = self._get_embedding_dim_dict()
    self._initialize_embeddings()

  @property
  def scope(self) -> str:
    return self._config.scope

  def _get_embedding_dim_dict(self) -> EmbedDimDict:
    """Get the feature-wise dimension of each embedding module."""
    embedding_dim_dict = {}
    for feat in self._config.identity_lookup_features:
      embedding_dim_dict[feat] = self._config.ndim_dict[feat]
    for feat in self._config.context_features + self._config.sequential_features:
      if feat not in embedding_dim_dict:
        embedding_dim_dict[feat] = self._config.ndim_emb
    return embedding_dim_dict

  def _initialize_embeddings(self):
    """Initializes the embeddings, depending on the embedding type."""
    with tf.variable_scope(self.scope):
      init_temporal_s = np.sqrt(
          6. / (self._config.nact_dict["num_s"] + self._config.ndim_emb + 1))

      self.w_dt = tf.get_variable(
          name="w_dt",
          shape=[1, self._config.ndim_emb],
          initializer=tf.initializers.random_uniform(
              -init_temporal_s, init_temporal_s))

      if self._config.embedding_type not in self._embedding_classes:
        raise ValueError(
            f"Unknown embedding type: {self._config.embedding_type}.")
      self.embedding = self._embedding_classes[self._config.embedding_type](
          self._config, self._embed_dim_dict)

  def get_total_embedding_size(self) -> Union[int, List[int]]:
    """Gets the total expected size of the embedding.

    Returns:
      Integer or List of Integers, the total size of the embedding. This
      depends on the features to be modelled and the encoder type.
    """
    features = self._config.context_features + self._config.sequential_features
    feature_dims = [self._embed_dim_dict[feat] for feat in features]
    # Concatenate all features, including upscaled time (dimension = ndim_emb)
    return sum(feature_dims) + self._config.ndim_emb


  def embed_batch(self, batch: batches.TFBatch) -> tf.Tensor:
    """Embeds input batch and creates inputs to temporal model."""
    data = batches.batch_to_components(batch, self._config.context_features,
                                       self._config.sequential_features)
    return self.embed_data(data)

  def embed_data(
      self,
      data: Dict[str, tf.SparseTensor]
      ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Embeds data dictionary and creates inputs to temporal model."""

    batch_shape = tf.shape(data["t"])[:-1]
    flat_data = nest.map_structure(batches.flatten_batch, data)
    flat_data = nest.map_structure(batches.sparse_fill_empty_rows, flat_data)

    context_embeddings = (
        self.embedding.provide_embeddings_to_forward_fn(
            flat_data, feature_types=self._config.context_features))
    context_embeddings = nest.map_structure(
        batches.get_unflatten_batch_fn(batch_shape), context_embeddings)

    sequential_embeddings = (
        self.embedding.provide_embeddings_to_forward_fn(
            flat_data, feature_types=self._config.sequential_features))
    sequential_embeddings = nest.map_structure(
        batches.get_unflatten_batch_fn(batch_shape), sequential_embeddings)

    dt = tf.divide(tf.cast(data["dt"], dtype=tf.float32), 5400.)
    t = tf.divide(tf.cast(data["t"], dtype=tf.float32), 5400.)
    dt_log = tf.log(dt + 1.)

    embedding_dict = sequential_embeddings.copy()
    embedding_dict.update(context_embeddings)
    embedding_dict["dt_s"] = tf.matmul(dt_log, self.w_dt)
    combined_embedding = self._combine_embeddings_for_input(embedding_dict)
    inputs = combined_embedding
    if self._config.get("apply_bias", False):
      inputs = inputs + tf.get_variable(
          "_".join([self._config.embedding_type, "final_bias"]),
          shape=[self.get_total_embedding_size()],
          initializer=tf.zeros_initializer)
    time_vect = t

    return inputs, time_vect

  def get_embedding_regularization_loss(self) -> tf.Tensor:
    """Gets the regularization loss for embedding weights.

    Returns:
      The embedding regularization loss. If
      embedding_regularize_only_sparse_lookup is true, regularization is only
      applied to the first layer of the embedding.
    """
    sparse_lookup_regularization = self._config.sparse_lookup_regularization
    sparse_lookup_regularization_weight = (
        self._config.sparse_lookup_regularization_weight)
    encoder_regularization = self._config.encoder_regularization
    encoder_regularization_weight = self._config.encoder_regularization_weight

    if self._config.get("embedding_regularize_only_sparse_lookup"):
      encoder_regularization_weight = 0.

    embedding_weights = [
        w for w in tf.trainable_variables() if (
            self._config.embedding_type.lower() in w.name.lower())]

    sparse_encoding_regularizer = model_utils.get_regularizer(
        sparse_lookup_regularization, sparse_lookup_regularization_weight)

    encoder_regularizer = model_utils.get_regularizer(
        encoder_regularization, encoder_regularization_weight)

    sparse_encoder_lookup_weights = [
        w for w in embedding_weights if "lookup" in w.name.lower()]
    encoder_weights = [
        w for w in embedding_weights if
        "lookup" not in w.name.lower()]

    if not sparse_encoding_regularizer:
      sparse_lookup_reg_penalty = tf.constant(0.)
    else:
      sparse_lookup_reg_penalty = slim.apply_regularization(
          sparse_encoding_regularizer, sparse_encoder_lookup_weights)

    if not encoder_regularizer or not encoder_weights:
      encoder_reg_penalty = tf.constant(0.)
    else:
      encoder_reg_penalty = slim.apply_regularization(encoder_regularizer,
                                                      encoder_weights)

    embedding_regularization_penalty = (
        sparse_lookup_reg_penalty + encoder_reg_penalty)

    return embedding_regularization_penalty

  def _combine_embeddings_for_input(
      self,
      embedding_dict: Dict[str, int]
      ) -> tf.Tensor:
    """Combines embeddings into one input for the model.

    The embeddings will be concatenated.

    Args:
      embedding_dict: dict of string feature name to embedding.
    Returns:
      The combined embeddings to be provided as an input for a particular step
      of the model.
    """
    return tf.concat(list(embedding_dict.values()), axis=-1)


# Type for any subclass of EncoderModule.
EncoderModuleType = TypeVar("EncoderModuleType", bound=EncoderModule)
