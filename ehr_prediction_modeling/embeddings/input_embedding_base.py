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
"""Base class for EHR embeddings."""

import abc
import collections
import typing
from typing import Any, Dict, List, Optional, Tuple

from ehr_prediction_modeling import types
import tensorflow.compat.v1 as tf

if typing.TYPE_CHECKING:
  from ehr_prediction_modeling import configdict


class InputEmbeddingBase(metaclass=abc.ABCMeta):
  """The base embedding class declaring the interface."""

  def __init__(self,
               encoder_config: "configdict.ConfigDict",
               emb_dim_dict: Dict[str, int],
               has_embedding_loss: Optional[bool] = False):
    """Initialize the embedding base object.

    Args:
      encoder_config: ConfigDict of encoder parameters.
      emb_dim_dict: Dict of feature name to dimension of the embedding for that
        feature.
      has_embedding_loss: True if this embedding has a loss.
    """
    self._config = encoder_config
    # Regularization depends on this
    self._name = self._config.embedding_type
    self._embed_dim_dict = emb_dim_dict
    # Boolean, whether in training mode to apply dropout / batch normalization.
    self._is_training = types.ModelMode.is_train(
        self._config.get("mode", types.ModelMode.TRAIN))
    self._all_features = self._config.context_features + self._config.sequential_features
    self.has_embedding_loss = has_embedding_loss
    self._encoders = {}

  def get_encoders(self):
    """Returns a dictionary from feature type to an encoder."""
    return self._encoders

  @abc.abstractmethod
  def _initialize_weights(self):
    """Initialize the weights for embedding computations.

       The derived classes are responsible for calling this function,
       so that additional inputs can be added to the class table if needed, and
       the order of construction can be controlled.
    """

  def get_embedding_loss(self, *unused_args, **unused_kwargs
                         ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Returns the total embedding loss.

    Returns:
      A tuple (total_embedding_loss, {feature_type: emb_loss_for_that_feature}).
    """
    return tf.constant(0, dtype=tf.float32), {}

  def _embed_feature(
      self,
      feat_type: str,
      input_data: Tuple[tf.SparseTensor, tf.SparseTensor],
      context: Optional[Dict[str, Any]] = None
      ) -> tf.Tensor:
    """Provides embedding for a single feature."""
    return self._encoders[feat_type](input_data, context)

  def provide_embeddings_to_forward_fn(
      self,
      data: Dict[str, tf.SparseTensor],
      feature_types: List[str]
      ) -> tf.Tensor:
    """Provides the embeddings to the forward function.

    Args:
      data: Dict of feature types to SparseTensors.
      feature_types: Names of feature types in data to embed.

    Returns:
      Dict of feature_types to embeddings.
    """
    embeddings = collections.OrderedDict()
    context = {"is_training": self._is_training}

    for feat_type in feature_types:
      input_data = (data[f"idx_{feat_type}"], data[f"val_{feat_type}"])

      embeddings[feat_type] = self._embed_feature(feat_type, input_data,
                                                  context)

    return embeddings
