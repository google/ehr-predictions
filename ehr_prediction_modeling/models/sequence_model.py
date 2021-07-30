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
"""Interface for sequence models."""

import abc
import typing
from typing import List, Optional, Union

from ehr_prediction_modeling import types
import sonnet as snt

import tensorflow.compat.v1 as tf

if typing.TYPE_CHECKING:
  from ehr_prediction_modeling import configdict




class SequenceModel(snt.AbstractModule, metaclass=abc.ABCMeta):
  """Interface for sequence models."""

  def __init__(self, config: "configdict.ConfigDict",
               embedding_size: Union[int, List[int]], name: str):
    """Initializes the parameters of the model.

    Args:
      config: Configuration specifying model parameters.
      embedding_size: The total size of the embedding.
      name: The name of the model
    """
    super().__init__(name=name)
    self._config = config
    self._embedding_size = embedding_size

  @abc.abstractmethod
  def _build(self, features: tf.Tensor) -> tf.Tensor:
    """Runs a forward pass through the model.

    Args:
      features: Tensor of shape [num_unroll, batch_size, embedding_size].

    Returns:
      Tensor of shape [num_unroll, batch_size, ndim_model_output].
    """

  @abc.abstractmethod
  def get_model_regularization_loss(self) -> tf.Tensor:
    """Gets the regularization loss for model weights."""

  @property
  def mode(self) -> str:
    """One of types.ModelMode."""
    return self._config.get("mode", types.ModelMode.TRAIN)

  @property
  def scope(self) -> str:
    """Scope for the RNN graph."""
    return self._config.scope

  @property
  def initial_state(self) -> Optional[List[tf.Tensor]]:
    """Returns initial state from all layers of the RNN.

    Not implemented for all RNN types and may return None.
    """
    return
