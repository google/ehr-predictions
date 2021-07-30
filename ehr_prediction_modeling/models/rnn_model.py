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
"""Definition of the RNN model class."""

import abc
from typing import List, Tuple, Union

from absl import logging
from ehr_prediction_modeling import types
from ehr_prediction_modeling.models import cell_factory
from ehr_prediction_modeling.models import model_utils
from ehr_prediction_modeling.models import sequence_model
from ehr_prediction_modeling.utils import activations as activation_util
import tensorflow.compat.v1 as tf
import tf_slim as slim
import tree

from ehr_prediction_modeling import configdict


class BaseRNNModel(sequence_model.SequenceModel, metaclass=abc.ABCMeta):
  """Base class defining common methods for RNN models."""

  MODEL_NAME = "rnn_model"

  def __init__(self, config: configdict.ConfigDict,
               embedding_size: Union[int, List[int]]) -> None:
    """Initializes the parameters of the model.

    Args:
      config: ConfigDict of model parameters.
      embedding_size: int or List[int], the total size of the embedding.

    Raises:
      ValueError: If the cell type is specified incorrectly.
    """
    super().__init__(config, embedding_size, self.MODEL_NAME)
    self._batch_size = self._config.batch_size
    self._num_unroll = self._config.num_unroll
    self._act_fn = activation_util.get_activation(
        act_fn=self._config.act_fn, lrelu_coeff=self._config.leaky_relu_coeff)
    self._depth = len(self._config.ndim_lstm)

    with self._enter_variable_scope():
      self._xavier_uniform_initializer = slim.xavier_initializer(
          uniform=True, seed=None, dtype=tf.float32)
      # Initialise the RNN. Note that the trainable variables of the RNN are not
      # created here but during the tf.contrib.rnn.static_state_saving_rnn call.
      self._init_rnn()
      if self._config.use_highway_connections:
        self._init_highway_size_adjustment_matrices()

  @abc.abstractmethod
  def _init_rnn(self) -> None:
    """Initializes the RNN."""

  def _init_highway_size_adjustment_matrices(self) -> None:
    if self._config.use_highway_connections:
      raise NotImplementedError(
          "use_highway_connections was set to True, but "
          "_init_highway_size_adjustment_matrices was not implemented for the "
          "RNN.")

  def _get_inputs_from_features(
      self, features: Union[List[tf.Tensor], tf.Tensor], t_vect: tf.Tensor,
      beginning_of_sequence_mask: tf.Tensor, prev_states: List[tf.Tensor]
  ) -> Tuple[Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor], tf.Tensor]:
    # Only set_shape if prev_states use static batch_size.
    if tree.flatten(prev_states)[0].shape[0].value is not None:
      if isinstance(features, list):
        for feats in features:
          feats.set_shape([self._num_unroll, self._batch_size, None])
      else:
        features.set_shape([self._num_unroll, self._batch_size, None])
      beginning_of_sequence_mask.set_shape([self._num_unroll, self._batch_size])
      if t_vect is not None:
        t_vect.set_shape([self._num_unroll, self._batch_size, None])

    return (features, tf.expand_dims(beginning_of_sequence_mask, -1))

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @batch_size.setter
  def batch_size(self, new_batch_size: int) -> None:
    self._batch_size = new_batch_size

  @property
  def num_unroll(self) -> int:
    return self._num_unroll

  @abc.abstractmethod
  def _get_variables_for_regularization(self) -> List[tf.Variable]:
    """Returns a list of variables to apply the regularizer on."""

  def get_model_regularization_loss(self) -> tf.Tensor:
    """See base class."""
    rnn_regularizer = model_utils.get_regularizer(
        self._config.l_regularization,
        tf.constant(self._config.l_reg_factor_weight))

    if rnn_regularizer:
      rnn_weights = [
          w for w in self._get_variables_for_regularization()
          if "cell" in w.name.lower()
      ]
      rnn_regularization_penalty = slim.apply_regularization(
          rnn_regularizer, rnn_weights)
    else:
      rnn_regularization_penalty = tf.constant(0.)

    return rnn_regularization_penalty


class RNNModel(BaseRNNModel):
  """Implementation of the RNN model."""

  def __init__(self, config: configdict.ConfigDict,
               embedding_size: Union[int, List[int]]) -> None:
    # If the SNREncoder was used then the embedding size will come as a list.
    embedding_size = (
        sum(embedding_size)
        if isinstance(embedding_size, list) else embedding_size)
    super().__init__(config, embedding_size)

  def _init_rnn(self) -> None:
    """Intializes the RNN."""
    def input_size_fn(layer_i):
      if self._config.use_highway_connections:
        return self._config.ndim_lstm[layer_i]
      else:
        return self._embedding_size if layer_i == 0 else (
            self._config.ndim_lstm)
    ith_val = lambda x, i: x[i] if isinstance(x, list) else x
    self._rnn = []
    for i, _ in enumerate(self._config.ndim_lstm):
      ith_cell_config = configdict.ConfigDict()
      ith_cell_config.cell_type = ith_val(self._config.cell_type, i)
      ith_cell_config.input_size = input_size_fn(i)
      ith_cell_config.ndim_lstm = ith_val(self._config.ndim_lstm, i)
      ith_cell_config.activation_fn = ith_val(self._act_fn, i)
      ith_cell_config.use_highway_connections = ith_val(
          self._config.use_highway_connections, i)
      for key, elt in self._config.cell_config.items():
        ith_cell_config[key] = elt
      logging.info("Generated config for RNN cell: \n%s", ith_cell_config)
      self._rnn.append(cell_factory.init_deep_cell(ith_cell_config))

  def _init_highway_size_adjustment_matrices(self) -> None:
    """Intializes the highway size adjustments."""
    highway_emb_shape = [
        self._embedding_size,
        (self._config.ndim_lstm[0] if isinstance(self._config.ndim_lstm, list)
         else self._config.ndim_lstm)]
    self._w_emb_to_lstm = tf.get_variable(
        "embedding_to_cell_weight",
        highway_emb_shape,
        tf.float32,
        self._xavier_uniform_initializer)
    self._w_cell_size_adjust = []
    if isinstance(self._config.ndim_lstm, list) and len(
        self._config.ndim_lstm) > 1:
      for index, (first_size, second_size) in enumerate(
          zip(self._config.ndim_lstm, self._config.ndim_lstm[1:])):
        if first_size != second_size:
          self._w_cell_size_adjust.append(tf.get_variable(
              "cell_size_adjust" + str(index),
              shape=[first_size, second_size],
              dtype="float32",
              initializer=self._xavier_uniform_initializer))
        else:
          self._w_cell_size_adjust.append(None)

  def initial_state(self,
                    batch_size: int,
                    dtype: tf.dtypes.DType = tf.float32) -> List[tf.Tensor]:
    """See base class."""
    return [rnn.initial_state(batch_size, dtype) for rnn in self._rnn]

  def _build(self, features: tf.Tensor, beginning_of_sequence_mask: tf.Tensor,
             t_vect: tf.Tensor,
             prev_states: List[tf.Tensor]) -> types.ForwardFunctionReturns:
    """Builds the forward graph and returns the logits.

    Args:
      features: Embedded batch of data, shape [num_unroll, batch_size,
        embedding_size].
      beginning_of_sequence_mask: Beginning of sequence mask.
      t_vect: Time data, shape [num_unroll, batch_size, 1]. Used for the phased
        LSTM cell only.
      prev_states: Previous RNN states.

    Returns:
      ForwardFunctionReturns, a tuple of predictions and state information.
    """
    if isinstance(features, list):
      features = tf.concat(features, axis=-1)

    if self._config.use_highway_connections:
      features = tf.matmul(features, self._w_emb_to_lstm)
    inputs = self._get_inputs_from_features(features, t_vect,
                                            beginning_of_sequence_mask,
                                            prev_states)

    lstm_outputs = []
    hidden_states = []

    # Stacked RNN.
    for depth_index in range(self._depth):
      with tf.variable_scope("rnn_layer" + str(depth_index)):
        outputs, state = tf.nn.dynamic_rnn(
            self._rnn[depth_index],
            inputs,
            time_major=True,
            dtype=tf.float32,
            initial_state=prev_states[depth_index],
            swap_memory=True,
            parallel_iterations=self._config.parallel_iterations,
        )
        lstm_outputs.append(outputs)
        hidden_states.append(state)

        if (self._config.use_highway_connections and
            (depth_index < self._depth - 1)):
          outputs = self._embed_inputs_for_highway_connections(
              outputs, depth_index)
        inputs = (outputs, tf.expand_dims(beginning_of_sequence_mask, -1))

    if self._config.get("dense_output", default=True):
      model_output = tf.concat(lstm_outputs, axis=-1)
      model_output.set_shape([None, None, sum(self._config.ndim_lstm)])
    else:
      model_output = lstm_outputs[-1]
      model_output.set_shape([None, None, self._config.ndim_lstm[-1]])

    # Ensure that the RNN trainable variables are in the
    # `TRAINABLE_VARIABLES` collection.
    # This is a hack to make Keras-style RNN cells work with TF-Replicator.
    for depth_index in range(self._depth):
      if hasattr(self._rnn[depth_index], "trainable_weights"):
        rnn_trainable_vars = self._rnn[depth_index].trainable_weights
        trainable_vars_collection = tf.get_collection_ref(
            tf.GraphKeys.TRAINABLE_VARIABLES)
        for rnn_var in rnn_trainable_vars:
          if rnn_var not in trainable_vars_collection:
            trainable_vars_collection.append(rnn_var)

    return types.ForwardFunctionReturns(
        model_output=model_output,
        hidden_states=hidden_states,
        inputs=None,
        activations=lstm_outputs)

  def _embed_inputs_for_highway_connections(self, inputs: tf.Tensor,
                                            depth_index: int) -> tf.Tensor:
    if self._w_cell_size_adjust[depth_index] is None:
      # Equivalent to right-multiplying by identity at each time step
      return inputs
    else:
      # Equivalent to right-multiplying by self._w_cell_size_adjust[depth_index]
      # at each time step
      return tf.matmul(inputs, self._w_cell_size_adjust[depth_index])

  def _get_variables_for_regularization(self) -> List[tf.Variable]:
    return list(self.get_all_variables())

  def extract_state(self, cell_type, state_list):
    """Returns the hidden state to inspect in the predictions."""
    extracted_state_list = []
    for state in state_list:
      if cell_type == types.RNNCellType.LSTM:
        cell_state, _ = tf.split(state, num_or_size_splits=2, axis=0)
        extracted_state_list.append(tf.squeeze(cell_state, 0))
      else:
        extracted_state_list.append(
            tf.zeros([self._batch_size, 1], dtype=tf.float32))
    return extracted_state_list
