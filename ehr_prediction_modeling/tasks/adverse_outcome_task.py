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
"""Utilities to handle EHR Adverse Outcome tasks in experiments."""
import enum
from typing import Dict, List, Mapping, Optional, Tuple, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.tasks import base_task
from ehr_prediction_modeling.tasks import task_data
from ehr_prediction_modeling.tasks import task_masks
from ehr_prediction_modeling.utils import batches
from ehr_prediction_modeling.utils import label_utils
from ehr_prediction_modeling.utils import loss_utils
from ehr_prediction_modeling.utils import mask_utils
import tensorflow.compat.v1 as tf

from ehr_prediction_modeling import configdict


class AdverseOutcomeLevels(enum.IntEnum):
  UNKNOWN = 0
  NONE = 1
  LEVEL_1 = 2
  LEVEL_2 = 3
  LEVEL_3 = 4


class AdverseOutcomeRisk(base_task.Task):
  """Task implementation for predictions risk of future Adverse Outcomes."""

  task_type = types.TaskTypes.ADVERSE_OUTCOME_RISK

  @property
  def default_masks(self) -> List[str]:
    return [
        # See Step 16 - Interval Censoring.
        mask_utils.INTERVAL_MASK,
        mask_utils.IGNORE_MASK,
        mask_utils.PADDED_EVENT_MASK
    ]

  @property
  def _supported_train_masks(self) -> Dict[str, List[str]]:
    """Map the names of masks available during training to their components."""
    return {
        **super()._supported_train_masks,
        **{
            task_masks.Train.SINCE_EVENT:
                self.default_masks + [
                    mask_utils.SINCE_EVENT_TRAIN_MASK
                ],
            "no_censored_patients": [
                mask_utils.PATIENT_MASK, mask_utils.IGNORE_MASK
            ]
        }
    }

  @property
  def _unformatted_supported_eval_masks(self) -> Dict[str, List[str]]:
    """See base class."""
    return {
        task_masks.Eval.BASE:
            self.default_masks,
        "no_censored_patients": [
            mask_utils.PATIENT_MASK,
            mask_utils.IGNORE_MASK,
        ],
        task_masks.Eval.SINCE_EVENT:
            self.default_masks + [
                mask_utils.SINCE_EVENT_EVAL_MASK,
            ],
    }

  def __init__(self, config: configdict.ConfigDict):
    self._lookahead_labels = [
        label_utils.get_adverse_outcome_lookahead_label_key(window_time)
        for window_time in config.window_times
    ]
    label_keys = list(self._lookahead_labels)
    if config.adverse_outcome_during_remaining_stay:
      self._during_stay_label = label_utils.ADVERSE_OUTCOME_IN_ADMISSION
      label_keys.append(self._during_stay_label)
    super().__init__(config, label_keys=label_keys)

  @property
  def num_targets(self) -> int:
    return len(self._label_keys)

  @property
  def target_names(self) -> List[str]:
    return self._label_keys

  @property
  def window_hours(self) -> List[int]:
    return self._config.window_times

  @property
  def prediction_task_type(self) -> str:
    return types.TaskType.BINARY_CLASSIFICATION


  def get_label_dicts(
      self
  ) -> Tuple[Dict[Optional[str], Union[
      tf.FixedLenSequenceFeature, tf.FixedLenFeature]], Dict[Union[
          str, None], Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]]]:
    """Gets the feature dictionaries to parse a tf.SequenceExample.

    These dicts are used in the parse op (either batched or non-batched):
    https://www.tensorflow.org/api_docs/python/tf/io/parse_single_sequence_example#args
    They should include any feature that is used by this task.

    Returns:
      context_d: Dictionaries of TF features to read in context.
      sequence_d: Dictionaries of TF features to read in sequence.
    """
    context_d, sequence_d = mask_utils.get_labels_for_masks(
        self._config.train_mask, self._config.eval_masks,
        self._all_supported_masks, self._config.time_since_event_label_key)

    for label_key in self._label_keys:
      sequence_d[label_key] = tf.FixedLenSequenceFeature([1], tf.int64)

    return context_d, sequence_d

  def _binary_targets(self, labels: tf.Tensor,
                      compare_value: int) -> tf.Tensor:
    """Extracts the binary target from the data.

    Args:
      labels: Tensor in time-major shape wnct [num_unroll, batch_size, channels,
        num_targets].
      compare_value: int to compare if label values are equal or greater than.

    Returns:
      Tensor, set to 1 for each entry greater or equal than the given value,
      in time-major shape wnct [num_unroll, batch_size, channels, num_targets].
    """
    compare_value = tf.constant(compare_value, dtype=tf.int64)
    binary_targets = tf.cast(tf.greater_equal(
        labels, compare_value), dtype=tf.float32)
    return binary_targets

  def get_targets(self, batch: batches.TFBatch) -> tf.Tensor:
    """Returns binary targets extracted from batch."""
    if (not self._lookahead_labels and
        not self._config.adverse_outcome_during_remaining_stay):
      raise ValueError(
          "No targets specified for Adverse Outcome task. Must specify either"
          " lookahead labels or adverse_outcome_during_remaining_stay.")

    if self._lookahead_labels:
      lookahead_labels = self._extract_labels(
          batch,
          self._lookahead_labels,
      )
      binary_targets = self._binary_targets(lookahead_labels,
                                            self._config.adverse_outcome_level)

    if self._config.adverse_outcome_during_remaining_stay:
      during_stay_labels = self._extract_labels(batch,
                                                [self._during_stay_label])
      during_stay_targets = self._binary_targets(during_stay_labels, 1)

      if self._lookahead_labels:
        binary_targets = tf.concat([binary_targets, during_stay_targets],
                                   axis=3)
      else:
        binary_targets = during_stay_targets

    return binary_targets

  def _get_all_task_variables(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.TaskVariables:
    """Computes variables for AdverseOutcomeRisk task.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      model_output: Tensor, the output from the model, shape wnt [num_unroll,
        batch_size, dim_model_output].

    Returns:
      task_data.TaskVariables with all the variables from this task.
    """
    logits = self.layer.get_logits(model_output)

    # This requires the targets to be sorted by window.
    if self._config.accumulate_logits:
      logits = tf.cumsum(logits, axis=-1)

    binary_targets = self.get_targets(batch)
    train_loss_mask = self.get_train_mask(batch)
    loss = loss_utils.loss_fn(
        logits,
        binary_targets,
        train_loss_mask,
        loss_type=types.TaskLossType.CE,
        scale_pos_weight=self._config.get("scale_pos_weight", 1.0))
    eval_mask_dict = self.get_eval_mask_dict(batch)
    eval_losses = self._get_eval_losses(logits, binary_targets, eval_mask_dict)

    predictions = tf.sigmoid(logits)
    return task_data.TaskVariables(
        loss=loss,
        targets=binary_targets,
        predictions=predictions,
        train_mask=train_loss_mask,
        eval_mask_dict=eval_mask_dict,
        eval_losses=eval_losses,
    )

  def _filter_by_target_names(self, tensor: tf.Tensor) -> tf.Tensor:
    if getattr(self._config, "eval_target_names", None):
      target_idx = [
          self.target_names.index(name)
          for name in self._config.eval_target_names
      ]
      target_mat = tf.one_hot(target_idx, len(self.target_names))
      tensor = tf.matmul(tensor, tf.transpose(target_mat))
    return tensor

  def get_eval_logits(self,
                      model_output: tf.Tensor) -> tf.Tensor:
    logits = self.layer.get_logits(model_output)
    # This requires the targets to be sorted by window.
    if self._config.accumulate_logits:
      logits = tf.cumsum(logits, axis=-1)

    eval_logits = self._filter_by_target_names(logits)
    return eval_logits

  def _get_eval_losses(self, logits: tf.Tensor,
                       binary_targets: tf.Tensor,
                       eval_mask_dict: Dict[str, tf.Tensor]):
    """Compute eval losses with respect to masks in eval_mask_dict.

    If self._config.eval_target_names is not empty, then losses are only
    computed with respect to targets specified in it.

    Args:
      logits: Tensor output from the model, expects shape [num_unroll,
        batch_size, num_targets].
      binary_targets: Tensor containing all targets, expects shapse [num_unroll,
        batch_size, num_targets].
      eval_mask_dict: Dict of mask names to masks.

    Returns:
      A dict of mask names to losses.
    """
    eval_losses = {}
    for eval_mask_name, eval_mask in eval_mask_dict.items():
      eval_logits = self._filter_by_target_names(logits)
      eval_binary_targets = self._filter_by_target_names(binary_targets)
      eval_loss_mask = self._filter_by_target_names(eval_mask)
      eval_losses[eval_mask_name] = loss_utils.loss_fn(
          eval_logits,
          eval_binary_targets,
          eval_loss_mask,
          loss_type=types.TaskLossType.CE)

    return eval_losses

  @classmethod
  def config(
      cls,
      window_times: List[int],
      adverse_outcome_level: AdverseOutcomeLevels,
      eval_masks: List[str],
      train_mask: str,
      adverse_outcome_during_remaining_stay: bool = False,
      time_since_event_hours_list: Optional[List[int]] = None,
      loss_weight: float = 1.0,
      accumulate_logits: bool = True,
      scale_pos_weight: Optional[float] = 1.,
      task_layer_type: str = types.TaskLayerTypes.MLP,
      task_layer_sizes: Optional[List[int]] = None,
      regularization_type: str = types.RegularizationType.NONE,
      regularization_weight: float = 0.,
      name: str = types.TaskNames.ADVERSE_OUTCOME_RISK,
      snr_config: Optional[configdict.ConfigDict] = None,
  ) -> configdict.ConfigDict:
    """Generates a config object for AKIRisk.

    Args:
      window_times: list of int, prediction windows for the adverse outcome.
      adverse_outcome_level: AdverseOutcomeLevels, sets the level
        of adverse outcome to be considered a positive target.
      eval_masks: list of str, names of the masks to be used in eval. The names
        should be in AdverseOutcomeRisk._supported_eval_masks.
      train_mask: str, name of the mask used for training. One of
        AdverseOutcomeRisk._supported_train_masks.
      adverse_outcome_during_remaining_stay: boolean indicating if the model
        should predict adverse outcome during the remainder of a patients in
        hospital stay.
      time_since_event_hours_list: If the since event mask is used, this list
        specifies the number of hours after event at which the model
        predicts Adverse Outcome during remaining stay e.g. [0, 6] would predict
        Adverse Outcome at event and 6 hours after event. If masking
        during eval, these are returned as separate masks to get metrics for
        predicting Adverse Outcome at event and Adverse Outcome at 6 hours
        after event.
      loss_weight: float, weight of this task loss.
      accumulate_logits: bool, whether to create a CDF over logits predict
        windows to encourage monotinicity.
      scale_pos_weight: float, weight of positive samples in the loss.
      task_layer_type: one of types.TaskLayerTypes - the type of layer to use.
      task_layer_sizes: array of int, the size of the task-specific layers to
        pass the model output through before a final logistic layer. If None,
        there is just the final logistic layer.
      regularization_type: one of types.RegularizationType, the regularization
        to be applied to the task layer(s).
      regularization_weight: float, the weight of the regularization penalty to
        apply to logistic layers associated with this task.
      name: str, name of this task for visualization and debugging.
      snr_config: configdict.ConfigDict, containing task layer sub-network
        routing parameters.

    Returns:
      A ConfigDict to be used to instantiate a AdverseOutcomeRisk task.
    """
    config = configdict.ConfigDict()
    config.task_type = AdverseOutcomeRisk.task_type
    config.name = name
    config.window_times = sorted(window_times)
    config.adverse_outcome_level = adverse_outcome_level
    config.eval_masks = eval_masks
    config.train_mask = train_mask
    config.adverse_outcome_during_remaining_stay = adverse_outcome_during_remaining_stay
    config.time_since_event_hours_list = time_since_event_hours_list or []
    # Since event label key used is time since admission.
    config.time_since_event_label_key = label_utils.TSA_LABEL
    config.loss_weight = loss_weight
    config.accumulate_logits = accumulate_logits
    config.scale_pos_weight = scale_pos_weight
    config.task_layer_sizes = task_layer_sizes or []
    config.regularization_type = regularization_type
    config.regularization_weight = regularization_weight
    # Non-empty tuple contain list of target names for computing eval loss.
    config.eval_target_names = ()
    return config

  @classmethod
  def default_configs(cls) -> List[configdict.ConfigDict]:
    """Generates a default config object for AdverseOutcomeRisk."""
    return [
        AdverseOutcomeRisk.config(
            window_times=label_utils.DEFAULT_LOOKAHEAD_WINDOWS,
            adverse_outcome_level=AdverseOutcomeLevels.LEVEL_1,
            train_mask="base_train",
            eval_masks=["base_eval"],
            scale_pos_weight=1.0,
            adverse_outcome_during_remaining_stay=False,
            loss_weight=1.0)
    ]
