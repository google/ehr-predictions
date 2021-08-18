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

"""Modeling layers used by tasks."""
import abc
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.models import model_utils
from ehr_prediction_modeling.models.nets import snr
from ehr_prediction_modeling.utils import activations
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import tf_slim as slim
import tree

from ehr_prediction_modeling import configdict

HiddenTaskLayerType = Union[snt.BatchApply, snr.SNRTaskLayer,
                           ]


class BaseTaskLayer(metaclass=abc.ABCMeta):
  """Base Class for dealing with task modeling layers."""

  def __init__(self, config: configdict.ConfigDict,
               num_task_targets: int) -> None:
    self._config = config
    self._task_layer = None
    self._num_targets = num_task_targets
    self._task_layer_sizes = self._config.get("task_layer_sizes", []).copy()
    self._regularization_type = self._config.get("regularization_type",
                                                 types.RegularizationType.NONE)
    self._regularization_weight = self._config.get("regularization_weight", 0.)
    self._init_task_layer()

  def _init_task_layer(self) -> None:
    """Initializes the fully connected task-specific layer of the model."""
    self._task_layer = self._get_task_layer()


  @property
  def layer_kwargs(self) -> Mapping[str, Any]:
    """Returns mapping of kwargs used for layer construction."""
    layers = self._task_layer_sizes + [self._num_targets]
    w_initializer = slim.layers.initializers.xavier_initializer(
        uniform=True, seed=None, dtype=tf.float32)
    kwargs = {
        "output_sizes":
            layers,
        "initializers": {
            "w": w_initializer,
            "b": tf.zeros_initializer
        },
        "activation":
            activations.get_activation(self._config.get("activation", "relu")),
        "name":
            self._config.name
    }
    return kwargs

  def get_hidden_layer(self) -> HiddenTaskLayerType:
    """Fetches the task layer this class manages."""
    return self._task_layer

  def get_regularization_loss(self) -> tf.Tensor:
    """Gets the regularization loss on task-specific layers."""
    regularizer = model_utils.get_regularizer(self._regularization_type,
                                              self._regularization_weight)
    if not regularizer:
      return tf.constant(0.)
    return slim.layers.regularizers.apply_regularization(
        regularizer, self._task_layer.trainable_variables)

  def _layer_logits(self, model_output: tf.Tensor) -> tf.Tensor:
    """Passes model output through task-specific layer to get logits."""
    return self._task_layer(model_output)

  def accumulate_logits(self, logits: tf.Tensor) -> tf.Tensor:
    """Sets all but first negative logits to zero and cumulative sums them."""
    # Create array of positive logits.
    zeros = tf.zeros_like(logits)
    positive_logits = tf.where(logits > 0, logits, zeros)
    # Create bool mask that only excludes first lookahead window.
    mask = tf.ones_like(logits)
    bool_mask = tf.greater(tf.cumsum(mask, axis=-1), 1)
    # Replace original logits with positive values over bool mask.
    transformed_logits = tf.where(bool_mask, positive_logits, logits)
    logits = tf.cumsum(transformed_logits, axis=-1)
    return logits

  def get_logits(self, model_output: tf.Tensor) -> tf.Tensor:
    """Passes model output through task-specific layers and formats output.

    Args:
      model_output: tensor of shape [num_unroll, batch_size, dim_model_output]
        or List[num_unroll, batch_size, dim_model_output] when using SNRNN as a
        model.

    Returns:
      Tensor of shape [num_unroll, batch_size, num_targets].
    """
    logits = self._layer_logits(model_output)
    num_unroll, batch_size, _ = tree.flatten(model_output)[0].shape
    logits.shape.assert_is_compatible_with(
        [num_unroll, batch_size, self._num_targets])

    # Reshape the tensor from wnt -> wnct [num_unroll, batch_size, num_targets]
    # -> [num_unroll, batch_size, channel, num_targets]
    logits = tf.expand_dims(logits, axis=2)

    if getattr(self._config, "accumulate_logits", False):
      logits = self.accumulate_logits(logits)

    return logits

  @abc.abstractmethod
  def _get_task_layer(self) -> HiddenTaskLayerType:
    """Returns the task layer hidden implementation."""
