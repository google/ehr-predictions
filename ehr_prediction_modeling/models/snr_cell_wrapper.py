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

"""Definition of SNR wrapper for RNN cells."""

from typing import List, Text, Tuple, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.models.nets import snr
import sonnet as snt
import tensorflow.compat.v1 as tf
from ehr_prediction_modeling import configdict


class SNRWrapper(snt.AbstractModule):
  """A wrapper around a ResetCore, part of a subnetwork routing architecture.

  This wrapper takes care of creating SNRConnections from each input in a list
  of inputs and feeding the combined result into the cell it wraps. In this
  architecture the cell will typically be a ResetCore.

  If the inputs passed are not a list, but a single tensor, then it will keep
  those as they are and pass them on.
  """

  def __init__(self,
               cell,
               cell_type: types.RNNCellType,
               tasks: List[Text],
               dropout_is_training: bool,
               snr_config: configdict.ConfigDict,
               parallel_iterations: int,
               should_route: bool = True,
               use_task_specific_routing: bool = True,
               connection_name: Text = 'routing'):
    super(SNRWrapper, self).__init__(name='snr_block')
    self._cell = cell
    self._cell_type = cell_type
    self._tasks = sorted(tasks)
    self._dropout_is_training = dropout_is_training
    self._snr_config = snr_config
    self._parallel_iterations = parallel_iterations
    self._should_route = should_route
    self._connection_name = connection_name
    self._snr_connections = []

  def _get_connection_name_for_task_and_cell(self, cell_index, task_name='all'):
    return f'{self._connection_name}{cell_index}_task{task_name}'

  def _get_inputs_from_list(self, inputs: List[tf.Tensor]) -> tf.Tensor:
    if not self._should_route:
      return tf.concat(inputs, axis=-1)

    routed_inputs = []
    for index, inp in enumerate(inputs):
      if self._snr_config.use_task_specific_routing and self._tasks:
        for task in self._tasks:
          routing = snr.SNRConnection(
              is_training=self._dropout_is_training,
              conn_config=self._snr_config,
              name=self._get_connection_name_for_task_and_cell(
                  cell_index=index, task_name=task))
          self._snr_connections.append(routing)
          routed_inputs.append(routing(inp))
      else:
        routing = snr.SNRConnection(
            is_training=self._dropout_is_training,
            conn_config=self._snr_config,
            name=self._get_connection_name_for_task_and_cell(cell_index=index))
        self._snr_connections.append(routing)
        routed_inputs.append(routing(inp))
    return snr.combine_inputs(routed_inputs,
                              self._snr_config.input_combination_method)

  def _build(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
             previous_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    if self._cell_type == types.RNNCellType.PLSTM:
      if isinstance(inputs[0][1], list):
        inputs_for_cell = ((inputs[0][0],
                            self._get_inputs_from_list(inputs[0][1])),
                           inputs[1])
      else:
        inputs_for_cell = inputs
    else:
      if isinstance(inputs[0], list):
        inputs_for_cell = (self._get_inputs_from_list(inputs[0]), inputs[1])
      else:
        inputs_for_cell = inputs

    return tf.nn.dynamic_rnn(
        self._cell,
        inputs=inputs_for_cell,
        time_major=True,
        dtype=tf.float32,
        initial_state=previous_state,
        swap_memory=True,
        parallel_iterations=self._parallel_iterations,
    )

  @property
  def trainable_weights(self) -> List[tf.Tensor]:
    """Trainable weights of the SNR cell."""
    return self._cell.trainable_weights

  def get_snr_connections(self) -> List[tf.Variable]:
    wrapper_variables = self.get_all_variables()
    return [
        variable for variable in wrapper_variables
        if self._connection_name.lower() in variable.name.lower()
    ]

  def get_connection_penalty(self) -> tf.Tensor:
    """Returns the combined penalty for all SNR connections in this wrapper."""
    return sum(
        [conn.get_regularization_loss() for conn in self._snr_connections])

  def initial_state(self, *args, **kwargs) -> List[tf.Tensor]:
    """Creates the initial state for the cell."""
    return self._cell.initial_state(*args, **kwargs)
