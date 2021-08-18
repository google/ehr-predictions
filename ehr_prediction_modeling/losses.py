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

"""Helper functions for running jobs."""

import collections
import itertools

from absl import logging
from ehr_prediction_modeling.models import snrnn_model
from ehr_prediction_modeling.models.nets import deep_encoders


def get_task_specific_snr_routing_connections(model, encoder, losses_per_task):
  """Returns the task-specific SNR connections.

  If use_task_specific_routing is disabled in both the model and the encoder
  it will return an (empty dictionary, empty list) tuple.

  For sub-network routing when optimizing task specific connections we
  want to optimize each corresponding to the specific task loss they're
  associated with.

  If the model type is not SNRNN and the encoder type is not SNR then this
  function will return empty collections.

  Args:
    model: The instantiated model. Must be an SNRNN in order to have
      any connections returned.
    encoder: The instantiated encoder. Must have SNREncoders associated
      with the embedding in order to have any connections returned.
    losses_per_task: A dictionary from task name to associated total loss for
      that task.

  Returns:
    A tuple where the first element is a dictionary from a task specific loss
    to a variable list that will be optimized wrt to it, and the second element
    is a set of all the routing connections in the architecture.
  """
  loss_to_vars = {}
  routing_connections_for_task = collections.defaultdict(set)
  # In some cases the model is wrapped in an RNNModelWithPersistentState
  # class.
  if (isinstance(model, snrnn_model.SNRNNModel) or
      isinstance(model.get_model, snrnn_model.SNRNNModel)):
    for task, conn in model.get_task_routing_connections().items():
      routing_connections_for_task[task].update(conn)
  for enc in encoder.embedding.get_encoders().values():
    if isinstance(enc, deep_encoders.SNREncoder):
      for task, conn in enc.get_task_routing_connections().items():
        routing_connections_for_task[task].update(conn)
  logging.info("Found routing connections for each task: %s",
               routing_connections_for_task)
  if routing_connections_for_task:
    for task_name, task_loss in losses_per_task.items():
      if routing_connections_for_task[task_name]:
        loss_to_vars[task_loss] = list(routing_connections_for_task[task_name])
  return loss_to_vars, list(
      itertools.chain.from_iterable(routing_connections_for_task.values()))


def get_loss_to_variables_dict(model, encoder, losses_per_task, all_variables,
                               total_loss):
  """Returns a dictionary from loss to vars_list to be used in the optimizer."""

  loss_to_vars, all_routing_connections = (
      get_task_specific_snr_routing_connections(
          model=model, encoder=encoder, losses_per_task=losses_per_task))
  # After the call above the loss_to_vars dict contains a mapping from
  # all routing variables to the corresponding loss to be used when updating
  # them during training.
  # If task routing is enabled it will be empty and we will update all
  # trainable variables wrt the total loss.
  all_routing_connection_names = [
      conn.name.lower() for conn in all_routing_connections
  ]
  non_routing_variables = [
      v for v in all_variables
      if v.name.lower() not in all_routing_connection_names
  ]
  loss_to_vars[total_loss] = non_routing_variables
  return loss_to_vars
