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
"""Utilities to handle EHR predictive task."""
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from absl import logging
from ehr_prediction_modeling import types
from ehr_prediction_modeling.tasks import base_task
from ehr_prediction_modeling.tasks import task_data
from ehr_prediction_modeling.utils import batches
from ehr_prediction_modeling.utils import loss_utils
import tensorflow.compat.v1 as tf

from ehr_prediction_modeling import configdict


class Coordinator():
  """A coordinator for working with multiple tasks."""

  def __init__(self, task_list: List[base_task.Task],
               optimizer_config: configdict.ConfigDict):
    """Constructor.

    Args:
      task_list: list of Task, tasks to be coordinated. regularization to be
        applied to task-specific layers.
      optimizer_config: ConfigDict of hyperparamters for the optimizer.
    """
    self._task_list = list(task_list)
    self._optimizer_config = optimizer_config
    if len(set(self.task_names)) != len(task_list):
      raise ValueError("Repeated task name is not allowed in task list: "
                       "{tasks}".format(tasks=self._task_list))
    sum_targets_for_all_tasks = sum(self.num_targets_list)
    logging.info("Total number of targets: %s", sum_targets_for_all_tasks)
    logging.info("List of all targets: %s", self.target_names_list)
    self._task_layers = {}
    for task in self._task_list:
      task_layer = task.get_hidden_layer()
      self._task_layers[task.name] = task_layer

  @property
  def num_targets_list(self) -> List[int]:
    return [task.num_targets for task in self._task_list]

  @property
  def task_list(self) -> List[base_task.Task]:
    return self._task_list

  @property
  def task_names(self) -> List[str]:
    return [task.name for task in self._task_list]

  @property
  def target_names_list(self) -> List[str]:
    return [task.target_names for task in self._task_list]

  @property
  def task_prediction_types(self) -> List[str]:
    return [task.prediction_task_type for task in self._task_list]

  @property
  def task_eval_masks_list(self) -> List[List[str]]:
    return [task.eval_masks for task in self._task_list]

  @property
  def task_layers(self) -> Mapping[str, Any]:
    return self._task_layers

  def get_task_layers_variables(self) -> Sequence[tf.Variable]:
    return sum([
        layer.get_all_variables()
        for layer in self._task_layers.values()
        if layer is not None
    ], ())

  def get_coordinator_variables(
      self, batch: batches.TFBatch,
      model_output: Union[tf.Tensor, None]) -> task_data.CoordinatorVariables:
    """Creates graph computation of targets, losses and masks for all tasks.

    Args:
      batch: Either tf.NextQueuedSequenceBatch, containing a batch of data. Or
        batches.TFBatch
      model_output: Either Tensor, the outputs from the model, shape nwt
        [seq_len, batch_size, ndim_model_output] Or None

    Returns:
      CoordinatorVariables with the combined loss (or None if model_output was
      None) and the list of TaskVariables for each task.
    """
    # necessary! "if <some_tensor>:" raises an exception
    if model_output is not None:
      return self._get_task_variables_and_loss(batch, model_output)
    else:
      return self._get_task_variables(batch)

  def get_task_regularization_losses(self) -> tf.Tensor:
    """Gets regularization loss for task-specific layers."""
    loss = tf.constant(0.)
    for task in self._task_list:
      loss += task.layer.get_regularization_loss()
    return loss

  def _get_task_variables_and_loss(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.CoordinatorVariables:
    """Creates graph computation of targets, losses and masks for all tasks.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      model_output: Tensor, the outputs from the model, shape nwt [seq_len,
        batch_size, ndim_model_output]

    Returns:
      CoordinatorVariables with the combined loss and the list of
      TaskVariables for each task.
    """
    task_variables_list = []
    task_losses = []
    task_eval_losses = []
    task_loss_weight_dict = {}
    for idx, task in enumerate(self._task_list):
      task_variables = task.get_task_variables(batch, model_output)
      task_loss = task_variables.loss
      eval_loss = sum(task_variables.eval_losses.values())
      loss_combo_type = self._optimizer_config.get(
          "task_loss_combination_type", types.LossCombinationType.SUM_ALL)
      if loss_combo_type == types.LossCombinationType.SUM_ALL:
        task_loss_weight = task.loss_weight
        weighted_task_loss = task_loss * task_loss_weight
        weighted_task_eval_loss = eval_loss * task_loss_weight
      elif loss_combo_type == types.LossCombinationType.UNCERTAINTY_WEIGHTED:
        # Compute uncertainty based multi-task loss weight based on
        # https://arxiv.org/abs/1705.07115.
        weighted_task_loss, weighted_task_eval_loss, task_loss_weight = (
            loss_utils.compute_uncertainty_multi_task_loss(
                task_loss, eval_loss, task.task_type,
                self._optimizer_config.sigma_min,
                self._optimizer_config.sigma_max,
                self._optimizer_config.max_loss_weight))
      else:
        raise ValueError("Unknown loss combination type. Choose one of"
                         "types.LossCombinationType.")
      task_variables_list.append(task_variables)
      task_losses.append(weighted_task_loss)
      task_eval_losses.append(weighted_task_eval_loss)
      task_loss_weight_dict[task.task_type + "_loss_weight"] = task_loss_weight
    combined_loss = tf.math.add_n(task_losses)
    combined_eval_loss = tf.math.add_n(task_eval_losses)

    return task_data.CoordinatorVariables(
        combined_loss=combined_loss,
        combined_eval_loss=combined_eval_loss,
        task_variables_list=task_variables_list,
        task_loss_weight_dict=task_loss_weight_dict)

  def _get_task_variables(
      self, batch: batches.TFBatch) -> task_data.CoordinatorVariables:
    """Extract targets and train_mask.

    When the model uses the sklearn API rather than the tf API, there is no such
    a thing as a computational graph. But we still need the targets and training
    mask to train the model.
    This method is very similar to get_coordinator_variables and allow to
    use the Task framework with a sklearn API model.

    Args:
      batch: a Batch object.

    Returns:
      CoordinatorVariables with TaskVariables for each task and combined_loss
      set to None.
    """
    task_variables_list = []
    for task in self._task_list:
      task_variables = task.get_task_variables(batch, None)
      task_variables_list.append(task_variables)

    return task_data.CoordinatorVariables(
        task_variables_list=task_variables_list,
        combined_loss=None,
        combined_eval_loss=None,
        task_loss_weight_dict={})

  def get_label_dicts(
      self
  ) -> Tuple[Dict[str, Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]],
             Dict[str, Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]]]:
    """Gets the feature dictionaries to parse a tf.SequenceExample.

    These dicts are used in the parse op (either batched or non-batched):
    https://www.tensorflow.org/api_docs/python/tf/io/parse_single_sequence_example#args
    They should include any feature that is used by any task.

    Returns:
      context_d: Dictionaries of TF features to read in context.
      sequence_d: Dictionaries of TF features to read in sequence.
    """
    context_d = {}
    sequence_d = {}
    for task in self._task_list:
      task_context_d, task_sequence_d = task.get_label_dicts()
      context_d.update(task_context_d)
      sequence_d.update(task_sequence_d)
    return context_d, sequence_d
