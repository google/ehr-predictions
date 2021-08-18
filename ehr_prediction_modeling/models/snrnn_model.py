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

"""Definition of the SNRNN model class."""

import collections
import itertools
from typing import Dict, List, Text, Union

from absl import logging
from ehr_prediction_modeling import types
from ehr_prediction_modeling.models import cell_factory
from ehr_prediction_modeling.models import model_utils
from ehr_prediction_modeling.models import rnn_model
from ehr_prediction_modeling.models import snr_cell_wrapper
import tensorflow.compat.v1 as tf
import tree

from ehr_prediction_modeling import configdict


class SNRNNModel(rnn_model.BaseRNNModel):
  """Implementation of the sub-network routing RNN model."""

  routing_connection_name = "model_routing"

  def __init__(self, config: configdict.ConfigDict, tasks: List[Text],
               embedding_size: int) -> None:
    """Initializes the parameters of the model.

    Args:
      config: ConfigDict of model parameters.
      tasks: The tasks we're training on.
      embedding_size: int, the total size of the embedding.

    Raises:
      ValueError: If the cell type is specified incorrectly.
    """
    self._tasks = sorted(tasks)
    self._rnn = []
    self._is_training = types.ModelMode.is_train(config.mode)
    super(SNRNNModel, self).__init__(config, embedding_size)
    # The sub-network routing stacked RNN model does not support highway
    # connections since there are no set successors for each cell.
    # Instead the sub-network routing connections can be extended to be used
    # between layers as well, instead of just adjacent layers.
    assert not self._config.use_highway_connections
    assert isinstance(self._config.ndim_lstm, list) and all(
        [isinstance(dims, list) for dims in self._config.ndim_lstm])
    self._widths = [len(dims) for dims in self._config.ndim_lstm]

  def input_size_fn(self, layer_i: int,
                    use_task_specific_routing: bool) -> Union[int, None]:

    def _compute_input_size_for_list(inputs):
      if (self._config.snr.input_combination_method ==
          types.SNRInputCombinationType.SUM_ALL):
        if not model_utils.all_elements_equal(inputs):
          raise ValueError(
              "Output of parallel embedding layers are of unequal shape."
              "Hence types.SNRInputCombinationType.SUM_ALL can't be "
              "applied. Debug or use any other method of combining "
              "inputs listed in types.SNRInputCombinationType.")
        return inputs[0]
      elif (self._config.snr.input_combination_method ==
            types.SNRInputCombinationType.CONCATENATE):
        return sum(inputs)
      else:
        raise ValueError(
            "Unknown SNR input combination type selected. Choose one of"
            "types.SNRInputCombinationType.")

    size_multiplier = 1
    if use_task_specific_routing and (
        self._config.snr.input_combination_method ==
        types.SNRInputCombinationType.CONCATENATE):
      size_multiplier = len(self._tasks)
    if layer_i == 0:
      # Note that if self._embedding_size is a list i.e. the output of the
      # SNREncoder, the SNRWrapper introduces a routing variable for each item
      # in the list and subsequently concatenates them. The concatenated input
      # is passed to the cell. Hence, the size of the input received by the RNN
      # cell is sum(self._embedding_size).
      if isinstance(self._embedding_size, list):
        return size_multiplier * _compute_input_size_for_list(
            self._embedding_size)
      else:
        return self._embedding_size
    else:
      return size_multiplier * _compute_input_size_for_list(
          self._config.ndim_lstm[layer_i - 1])

  def _init_rnn(self) -> None:
    """Initializes the RNN."""
    self._rnn = []
    ith_val = lambda x, i: x[i] if isinstance(x, list) else x
    for i, widths in enumerate(self._config.ndim_lstm):
      self._rnn.append([])
      for j, _ in enumerate(widths):
        ijth_cell_config = configdict.ConfigDict()
        ijth_cell_config.mode = ith_val(ith_val(self._config.mode, i), j)
        ijth_cell_config.cell_type = ith_val(
            ith_val(self._config.cell_type, i), j)
        ijth_cell_config.input_size = self.input_size_fn(
            layer_i=i,
            use_task_specific_routing=self._config.snr.get(
                "use_task_specific_routing", False))
        ijth_cell_config.ndim_lstm = ith_val(
            ith_val(self._config.ndim_lstm, i), j)
        ijth_cell_config.activation_fn = ith_val(ith_val(self._act_fn, i), j)
        ijth_cell_config.use_highway_connections = False
        for key, elt in self._config.cell_config.items():
          ijth_cell_config[key] = elt
        snr_cell = snr_cell_wrapper.SNRWrapper(
            cell=cell_factory.init_deep_cell(ijth_cell_config),
            cell_type=self._config.cell_type,
            tasks=self._tasks,
            dropout_is_training=self._is_training,
            snr_config=self._config.snr,
            parallel_iterations=self._config.parallel_iterations,
            use_task_specific_routing=(
                self._config.snr.use_task_specific_routing),
            connection_name=SNRNNModel.routing_connection_name)
        logging.info("Generated config for SNRNN cell: \n%s", ijth_cell_config)
        self._rnn[-1].append(snr_cell)

  def initial_state(
      self,
      batch_size: int,
      dtype: tf.dtypes.DType = tf.float32) -> List[List[tf.Tensor]]:
    return [[rnn.initial_state(batch_size, dtype)
             for rnn in rnns]
            for rnns in self._rnn]

  def _build(
      self, features: tf.Tensor, beginning_of_sequence_mask: tf.Tensor,
      t_vect: tf.Tensor,
      prev_states: List[List[tf.Tensor]]) -> types.ForwardFunctionReturns:
    """Builds the forward graph and returns the logits.

    Args:
      features: embedded batch of data, shape [num_unroll, batch_size,
        embedding_size]
      beginning_of_sequence_mask: Beginning of sequence mask.
      t_vect: time data, shape [num_unroll, batch_size, 1]. Used for the phased
        LSTM cell only.
      prev_states: Previous RNN states.

    Returns:
      ForwardFunctionReturns tuple of predictions and state information.
    """
    inputs = self._get_inputs_from_features(features, t_vect,
                                            beginning_of_sequence_mask,
                                            prev_states)

    lstm_outputs = [
        [None for j in range(self._widths[i])] for i in range(self._depth)
    ]
    hidden_states = [
        [None for j in range(self._widths[i])] for i in range(self._depth)
    ]

    # Stacked RNN using sub-network routing.
    for depth_index in range(self._depth):
      with tf.variable_scope("rnn_layer" + str(depth_index)):
        for width_index in range(self._widths[depth_index]):
          with tf.variable_scope("rnn_width_layer" + str(width_index)):
            outputs, state = self._rnn[depth_index][width_index](
                inputs, prev_states[depth_index][width_index])
            lstm_outputs[depth_index][width_index] = outputs
            hidden_states[depth_index][width_index] = state
        if self._config.cell_type == types.RNNCellType.PLSTM:
          inputs = ((t_vect, lstm_outputs[depth_index]),
                    tf.expand_dims(beginning_of_sequence_mask, -1))
        else:
          inputs = (lstm_outputs[depth_index],
                    tf.expand_dims(beginning_of_sequence_mask, -1))

    if self._config.snr.get("should_pass_all_cell_outputs", True):
      activations = list(itertools.chain(*lstm_outputs))
    else:
      activations = lstm_outputs[-1]

    model_output = []
    for output in activations:
      output.set_shape([None, None, output.shape[2]])
      model_output.append(output)

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
        activations=activations)

  def _get_snr_connections(self) -> List[tf.Variable]:
    snr_connections = list(
        itertools.chain.from_iterable(
            [cell.get_snr_connections() for cell in tree.flatten(self._rnn)]))
    return snr_connections

  def _get_variables_for_regularization(self) -> List[tf.Variable]:
    snr_connections = self._get_snr_connections()
    return [
        var for var in self.get_all_variables() if var not in snr_connections
    ]

  def get_task_routing_connections(self) -> Dict[str, List[tf.Variable]]:
    """Returns a dictionary of task specific routing connections.

    If use_task_specific_routing is set to false then the dictionary will be
    empty as no task specific routing connections were created.
    """
    if not self._config.snr.use_task_specific_routing:
      return {}

    snr_connections = self._get_snr_connections()
    all_routing_connections_per_task = collections.defaultdict(list)
    for task_name in self._tasks:
      for w in snr_connections:
        if f"task{task_name.lower()}" in w.name.lower():
          all_routing_connections_per_task[task_name].append(w)
    return all_routing_connections_per_task

  def get_snr_regularization_loss(self) -> tf.Tensor:
    """Gets the regularization loss for all the routing connections."""
    return sum([
        snr_cell.get_connection_penalty()
        for snr_cell in tree.flatten(self._rnn)
    ])

  def get_model_regularization_loss(self) -> tf.Tensor:
    """Gets the regularization loss for model weights and connections."""
    base_model_regularization_loss = super().get_model_regularization_loss()
    snr_connections_penalty = self.get_snr_regularization_loss()
    return base_model_regularization_loss + snr_connections_penalty
