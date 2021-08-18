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
import abc
import copy
from typing import Any, Dict, List, Mapping, Optional, Union

from ehr_prediction_modeling import mask_manager
from ehr_prediction_modeling.tasks import mlp_task_layer
from ehr_prediction_modeling.tasks import task_data
from ehr_prediction_modeling.tasks import task_layers
from ehr_prediction_modeling.tasks import task_masks
from ehr_prediction_modeling.utils import batches
from ehr_prediction_modeling.utils import mask_utils
import tensorflow.compat.v1 as tf
from ehr_prediction_modeling import configdict


class Task(metaclass=abc.ABCMeta):
  """Interface for dealing with tasks."""
  task_type = ""

  def __init__(self, config, label_keys=None):
    self._config = config
    self.mask_manager = None
    self._config.eval_masks = self._update_eval_mask_names_list(
        self._config.eval_masks)
    self._label_keys = label_keys if label_keys else []
    self._task_layer = task_layers.get_task_layer(config, self.num_targets)
    self._init_mask_manager()

  def _init_mask_manager(self):
    self.mask_manager = mask_manager.MaskManager(
        task_config=self._config,
        label_keys=self._label_keys,
        window_hours=self.window_hours,
        supported_train_masks=self._supported_train_masks,
        supported_eval_masks=self._supported_eval_masks,
    )

  @property
  @abc.abstractmethod
  def default_masks(self) -> List[str]:
    """A list of masks that are used in all this task's composite masks."""

  @property
  def _supported_train_masks(self) -> Dict[str, List[str]]:
    return {
        task_masks.Train.BASE:
            self.default_masks,
    }

  @property
  @abc.abstractmethod
  def _unformatted_supported_eval_masks(self) -> Mapping[str, List[str]]:
    """Returns mapping of supported eval masks without task_type prepended..

    Returns:
      Map the names of masks (without task_type prepended) available during
      evaluation to their components.
    """

  def _update_eval_mask_names_list(self, eval_masks: List[str]) -> List[str]:
    """Updates eval mask names to include type and expand hours since event.

    Args:
      eval_masks: Eval masks to update the names of.

    Returns:
      An updated list of eval mask names. See _update_eval_mask_names Returns
      for a complete description of how mask names are updated.
    """
    # Convert to a dict so the same update fn can be applied to a list.
    empty_values = [None] * len(eval_masks)
    eval_mask_dict = dict(zip(eval_masks, empty_values))
    updated_eval_masks = self._update_eval_mask_names(eval_mask_dict)
    return list(updated_eval_masks)

  def _update_eval_mask_names(self, eval_masks: Dict[str, Optional[List[str]]]):
    """Updates eval mask names to include type and expand hours since event.

    Args:
      eval_masks: Eval masks to update the names of.

    Returns:
      A dict of eval masks with the keys updated. Keys will have task_type added
      and any mask with 'since_event_eval' in the name will be expanded based on
      time_since_event_hours_list. For example, if since_event_eval mask is a
      key in eval_masks and config.time_since_event_hours_list = [24, 48]. The
      resulting dict will have one entry for 24 hours after event and one
      entry for 48 hours after event, but no entry for the bare
      since_event_eval mask. If config.time_since_event_hours_list is empty, any
      since_event_eval mask will be removed.
    """
    since_event_masks = [
        mask_name for mask_name in eval_masks.keys()
        if task_masks.Eval.SINCE_EVENT in mask_name
    ]
    for mask_name in since_event_masks:
      for hours in self._config.get("time_since_event_hours_list", []):
        new_mask_name = mask_name.replace(
            task_masks.Eval.SINCE_EVENT,
            f"{hours}_{mask_utils.SINCE_EVENT_MASK_SUFFIX}")
        eval_masks[new_mask_name] = eval_masks[mask_name]
      del eval_masks[mask_name]

    return {
        mask_utils.get_unique_mask_name(self.task_type, mask_name): components
        for mask_name, components in eval_masks.items()
    }

  @property
  def _supported_eval_masks(self) -> Dict[str, List[str]]:
    """Returns mapping of all supported eval masks with task_type prepended.

    Expands since event masks to have one entry per time_since_event_hours_list.

     Returns:
      Map the names of masks (with task_type prepended) available during
      evaluation to their components.
    """
    unformatted_eval_masks = self._unformatted_supported_eval_masks
    return self._update_eval_mask_names(unformatted_eval_masks)

  @property
  def _all_supported_masks(self) -> Dict[str, List[str]]:
    masks = copy.copy(self._supported_train_masks)
    masks.update(self._supported_eval_masks)
    return masks


  @property
  def layer(self) -> task_layers.TaskLayers:
    return self._task_layer

  @property
  def name(self) -> str:
    return self._config.name

  @property
  @abc.abstractmethod
  def prediction_task_type(self) -> str:
    """Returns one of the values defined in {types.TaskType}."""

  @abc.abstractmethod
  def get_label_dicts(
      self) -> Dict[str, Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]]:
    """Returns a dictionary of labels to tensors that are used for the task."""

  @property
  @abc.abstractmethod
  def num_targets(self) -> int:
    """Total number of targets for the task."""

  @property
  @abc.abstractmethod
  def target_names(self) -> List[str]:
    """Names of targets for the task."""


  @property
  @abc.abstractmethod
  def window_hours(self) -> List[int]:
    """The total number of time horizons.

    Note that this list possibly contains dupplicated values, e.g. with the Labs
    task. It is because there are several labs with the same time horizons,
    corresponding to several different targets that may have different mask
    values (see mask_utils.TIME_CUTOFF_MASK).

    Returns:
      A list of the time horizons (in hours) of all the targets. If several
      targets have the same time horizon, the values are duplicated.
    """

  @abc.abstractmethod
  def _get_all_task_variables(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.TaskVariables:
    """Fetches all variables used by the task."""

  def _get_targets_and_masks(self,
                             batch: batches.TFBatch) -> tf.Tensor:
    targets = self.get_targets(batch)
    train_loss_mask = self.get_train_mask(batch)
    eval_mask_dict = self.get_eval_mask_dict(batch)
    return task_data.TaskVariables(
        targets=targets,
        train_mask=train_loss_mask,
        eval_mask_dict=eval_mask_dict,
    )

  def get_task_variables(
      self, batch: batches.TFBatch,
      model_output: Union[tf.Tensor, None]) -> tf.Tensor:
    """Computes variables for task.

    Args:
      batch: Either tf.NextQueuedSequenceBatch, containing a batch of data. Or
        batches.TFBatch
      model_output: Either Tensor, the output from the model, shape wnt [
        num_unroll, batch_size, ndim_model_output]. Or None

    Returns:
      task_data.TaskVariables with all the variables from this task.
    """
    if model_output is not None:
      return self._get_all_task_variables(batch, model_output)
    else:
      return self._get_targets_and_masks(batch)

  def get_targets(self, batch: batches.TFBatch) -> tf.Tensor:
    return self._extract_labels(batch, self._label_keys)

  def get_train_mask(self, batch: batches.TFBatch) -> tf.Tensor:
    """Computes the mask to be used to mask the training loss.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.

    Returns:
      Tensor, the loss mask to be used in training, in time-major
      shape wnct [num_unroll, batch_size, channels, num_targets].
    """
    train_mask = self._config.get("train_mask", task_masks.Train.BASE)
    if train_mask not in self._supported_train_masks:
      raise ValueError(
          "Train mask {mask} is not supported".format(mask=train_mask))

    return self.mask_manager.get_masks([train_mask], batch)[train_mask]

  def get_eval_mask_dict(
      self, batch: batches.TFBatch) -> Dict[str, tf.Tensor]:
    """Computes the dict of loss masks to be used to mask evaluation.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.

    Returns:
      dict of string mask name to Tensors, the loss masks to be used in
      evaluation, in time-major shape wnct [num_unroll, batch_size, channels,
      num_targets].

    """
    for eval_mask in self._config.eval_masks:
      if eval_mask not in self._supported_eval_masks:
        raise ValueError(
            "Eval mask {mask} is not supported".format(mask=eval_mask))
    return self.mask_manager.get_masks(
        self._config.eval_masks,
        batch,
    )

  @property
  def loss_weight(self) -> float:
    return self._config.loss_weight

  @property
  def eval_masks(self) -> List[str]:
    return self._config.eval_masks

  @property
  def task_layer_sizes(self) -> List[int]:
    return self._config.get("task_layer_sizes", []).copy()

  def _extract_labels(self, batch: batches.TFBatch,
                      label_keys: List[str]) -> tf.Tensor:
    """Extracts the labels denoted by label_keys from the data.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      label_keys: list of keys used to extract labels from the batch.

    Returns:
      Tensor in time-major shape wnct
      [num_unroll, batch_size, channels, num_targets] with the labels for each
      key given in label_keys.
    """
    return tf.stack(
        [batch.sequences[label_key] for label_key in label_keys], axis=3)

  def get_hidden_layer(self) -> mlp_task_layer.HiddenTaskLayerType:
    """Returns the underlying modeling layer from this tasks layer."""
    return self.layer.get_hidden_layer()

  # pytype: disable=bad-return-type
  @classmethod
  @abc.abstractmethod
  def config(cls) -> configdict.ConfigDict:
    """Config creation for the task."""

  # pytype: enable=bad-return-type

  # pytype: disable=bad-return-type
  @classmethod
  @abc.abstractmethod
  def default_configs(cls) -> List[configdict.ConfigDict]:
    """Default task config."""

  # pytype: enable=bad-return-type
