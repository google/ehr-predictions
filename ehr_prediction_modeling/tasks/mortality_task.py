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
"""Mortality task implementation to be used in experiments."""

from typing import Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
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


class MortalityRisk(base_task.Task):
  """Task implementation for mortality risk."""

  task_type = types.TaskTypes.MORTALITY

  @property
  def default_masks(self) -> List[str]:
    return [mask_utils.IGNORE_MASK, mask_utils.PADDED_EVENT_MASK]

  @property
  def _supported_train_masks(self) -> Dict[str, List[str]]:
    return {
        **super()._supported_train_masks,
        **{
            task_masks.Train.SINCE_EVENT:
                self.default_masks + [mask_utils.SINCE_EVENT_TRAIN_MASK],
        }
    }

  @property
  def _unformatted_supported_eval_masks(self) -> Dict[str, List[str]]:
    """See base class."""
    return {
        task_masks.Eval.BASE:
            self.default_masks,
        task_masks.Eval.SINCE_EVENT:
            self.default_masks + [mask_utils.SINCE_EVENT_EVAL_MASK],
    }

  def __init__(self, config: configdict.ConfigDict):
    label_keys = self._target_names = [
        label_utils.MORTALITY_LOOKAHEAD_LABEL_BASE.format(
            hours=str(hours)) for hours in self.window_hours
    ]
    if config.mortality_during_admission:
      label_keys.append(label_utils.MORTALITY_IN_ADMISSION_LABEL)
      self._target_names.append(label_utils.MORTALITY_IN_ADMISSION_LABEL)
    logging.info("Targets are: %s", str(self._target_names))
    super().__init__(config, label_keys=label_keys)

  @property
  def num_targets(self) -> int:
    return len(self._label_keys)

  @property
  def target_names(self) -> List[str]:
    return self._target_names

  @property
  def window_hours(self) -> List[int]:
    return [d * 24 for d in self._config.window_days]

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

    sequence_d[label_utils.IGNORE_LABEL] = tf.FixedLenSequenceFeature([1],
                                                                      tf.int64)
    for label_key in self._label_keys:
      sequence_d[label_key] = tf.FixedLenSequenceFeature([1], tf.int64)

    return context_d, sequence_d

  def _get_all_task_variables(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.TaskVariables:
    """Computes variables for MortalityRisk task.

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

    targets = self.get_targets(batch)
    train_loss_mask = self.get_train_mask(batch)

    loss = loss_utils.loss_fn(
        logits,
        targets,
        train_loss_mask,
        scale_pos_weight=self._config.get("scale_pos_weight", 1.0),
        loss_type=types.TaskLossType.CE)
    eval_mask_dict = self.get_eval_mask_dict(batch)
    predictions = tf.sigmoid(logits)

    return task_data.TaskVariables(
        loss=loss,
        targets=targets,
        predictions=predictions,
        train_mask=train_loss_mask,
        eval_mask_dict=eval_mask_dict,
    )

  @classmethod
  def config(
      cls,
      window_days: List[int],
      eval_masks: List[str],
      train_mask: str,
      time_buckets_per_day: int,
      mortality_during_admission: bool = False,
      time_since_event_hours_list: Optional[List[int]] = None,
      accumulate_logits: bool = True,
      loss_weight: float = 1.0,
      scale_pos_weight: Optional[float] = 1.,
      task_layer_type: str = types.TaskLayerTypes.MLP,
      task_layer_sizes: Optional[List[int]] = None,
      regularization_type: str = types.RegularizationType.NONE,
      regularization_weight: float = 0.,
      name: str = types.TaskNames.MORTALITY,
      snr_config: Optional[configdict.ConfigDict] = None,
  ) -> configdict.ConfigDict:
    """Generates a config object for MortalityRisk.

    Args:
      window_days: list of int, prediction windows in days for the mortality
        task. These should be present in the etl version used. In most etl
        versions, values from label_utils.DEFAULT_MORTALITY_WINDOWS are used.
      eval_masks: list of str, names of the masks to be used in eval. The names
        should be in MortalityRisk._supported_eval_masks.
      train_mask: str, name of the mask used for training. One of
        MortalityRisk._supported_train_masks.
      time_buckets_per_day: Number of time buckets within a day. It is only
        needed for time since event masks.
      mortality_during_admission: boolean indicating if the model should predict
        mortality during the remainder of a patient's in-hospital stay.
      time_since_event_hours_list: if the since event mask is present in either
        train or eval, this is a list specifying the number of hours after
        event at which the model predicts mortality e.g. [24, 48] would
        predict mortality at 24 and 48 hours after event. If masking during
        training, these are combined i.e. the loss is comprised of the
        prediction at 24 and  48 hours after event. If masking during eval,
        these are returned as separate masks to get metrics for predicting
        mortality at 24 hours after event, and separately for predicting
        mortality at 48 hours after event.
      accumulate_logits: bool, whether to create a CDF over the logits of
        increasing time_windows to encourage monotonicity.
      loss_weight: float, weight of this task loss.
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
      A ConfigDict to be used to instantiate a Mortality task.
    """
    config = configdict.ConfigDict()
    config.task_type = MortalityRisk.task_type
    config.name = name
    config.window_days = sorted(window_days)
    config.time_since_event_hours_list = time_since_event_hours_list or []
    # Since event label key used is time since admission.
    config.time_since_event_label_key = label_utils.TSA_LABEL
    config.time_buckets_per_day = time_buckets_per_day
    config.eval_masks = eval_masks
    config.train_mask = train_mask
    config.mortality_during_admission = mortality_during_admission
    config.accumulate_logits = accumulate_logits
    config.scale_pos_weight = scale_pos_weight
    config.loss_weight = loss_weight
    config.task_layer_sizes = task_layer_sizes or []
    config.regularization_type = regularization_type
    config.regularization_weight = regularization_weight
    return config

  @classmethod
  def default_configs(cls) -> List[configdict.ConfigDict]:
    """Generates a default config object for MortalityRisk."""
    return [
        MortalityRisk.config(
            window_days=[1, 7, 30],
            train_mask=task_masks.Train.BASE,
            time_buckets_per_day=24,
            eval_masks=[task_masks.Eval.BASE],
            mortality_during_admission=False,
            time_since_event_hours_list=None,
            scale_pos_weight=1.0,
            loss_weight=1.0,)
    ]
