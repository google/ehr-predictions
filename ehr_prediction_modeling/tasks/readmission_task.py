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
"""Readmission task implementation to be used in experiments."""
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


class ReadmissionRisk(base_task.Task):
  """Task implementation for readmission risk."""

  task_type = types.TaskTypes.READMISSION

  @property
  def default_masks(self) -> List[str]:
    return [
        mask_utils.IGNORE_MASK, mask_utils.AT_DISCHARGE_MASK,
        mask_utils.PADDED_EVENT_MASK
    ]

  @property
  def _unformatted_supported_eval_masks(self) -> Dict[str, List[str]]:
    """See base class."""
    return {task_masks.Eval.BASE: self.default_masks}

  def __init__(self, config: configdict.ConfigDict):
    if not config.binarize_days:
      raise ValueError(
          "You should provide a list of integers for `config.binarize_days` but"
          " it was empty.")
    label_keys = label_utils.get_readmission_label_keys(
        [int(w) for w in config.binarize_days])
    super().__init__(config, label_keys=label_keys)

  @property
  def num_targets(self) -> int:
    return len(self._label_keys)

  @property
  def target_names(self) -> List[str]:
    return self._label_keys

  @property
  def window_hours(self) -> List[int]:
    return [d * 24 for d in self._config.binarize_days]

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
        self._all_supported_masks)
    sequence_d[label_utils.TIME_UNTIL_NEXT_ADMISSION] = (
        tf.FixedLenSequenceFeature([1], tf.float32))
    return context_d, sequence_d

  def _binarized_readmission_label(
      self, binarize_days: List[int],
      batch: batches.TFBatch) -> tf.Tensor:
    """Gets the binarized Readmission labels for a given batch.

    Args:
      binarize_days: list of integers, the days around which to binarize
        readmission.
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.

    Returns:
      Tensor in time-major shape wnct [num_unroll, batch_size,
      channels, num_targets], with value 1 if the next admission is within the
      given time period (defined by binarize_days), and 0 otherwise
    """
    binarized_labels = []
    for day in binarize_days:
      binarized_batch_labels = tf.less_equal(
          batch.sequences[label_utils.TIME_UNTIL_NEXT_ADMISSION],
          tf.constant(day, dtype=tf.float32))
      binarized_labels.append(binarized_batch_labels)
    binarized_labels_concat = tf.stack(binarized_labels, axis=3)
    return tf.cast(binarized_labels_concat, tf.float32)

  def get_targets(
      self,
      batch: batches.TFBatch,
  ) -> tf.Tensor:
    return self._binarized_readmission_label(self._config.binarize_days, batch)

  def _get_all_task_variables(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.TaskVariables:
    """Computes variables for ReadmissionRisk task.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      model_output: Tensor, the output from the model, shape wnt [num_unroll,
        batch_size, dim_model_output].

    Returns:
      base_task.TaskGraphVariables with all the variables from this task.
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
        loss_type=types.TaskLossType.CE,
        scale_pos_weight=self._config.get("scale_pos_weight", 1.0))
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
      binarize_days: List[int],
      eval_masks: List[str],
      train_mask: str,
      accumulate_logits: bool = True,
      loss_weight: float = 1.0,
      discard_final_admission: bool = False,
      scale_pos_weight: float = 1.0,
      task_layer_type: str = types.TaskLayerTypes.MLP,
      task_layer_sizes: Optional[List[int]] = None,
      regularization_type: str = types.RegularizationType.NONE,
      regularization_weight: float = 0.,
      name: str = types.TaskNames.READMISSION,
      snr_config: Optional[configdict.ConfigDict] = None,
  ) -> configdict.ConfigDict:
    """Generates a config object for ReadmissionRisk.

    Args:
      binarize_days: list of int, prediction windows in days for the readmission
        task.
      eval_masks: list of str, names of the masks to be used in eval. The names
        should be in ReadmissionRisk._supported_eval_masks.
      train_mask: str, name of the mask used for training. One of
        ReadmissionRisk._supported_train_masks.
      accumulate_logits: bool, whether to create a CDF over the logits of
        increasing time_windows to encourage monotonicity.
      loss_weight: float, weight of this task loss.
      discard_final_admission: bool. Whether to mask out labels 2 (i.e. the
        final event of an admission that is also the last episode).
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
      A ConfigDict to be used to instantiate a Readmission task.
    """
    config = configdict.ConfigDict()
    config.task_type = ReadmissionRisk.task_type
    config.name = name
    config.binarize_days = sorted(binarize_days)
    config.eval_masks = eval_masks
    config.train_mask = train_mask
    config.accumulate_logits = accumulate_logits
    config.loss_weight = loss_weight
    config.discard_final_admission = discard_final_admission
    config.scale_pos_weight = scale_pos_weight
    config.task_layer_sizes = task_layer_sizes or []
    config.regularization_type = regularization_type
    config.regularization_weight = regularization_weight
    config.time_since_event_hours_list = []  # unused, here for consistency
    return config

  @classmethod
  def default_configs(cls) -> List[configdict.ConfigDict]:
    """Generates a default config object for ReadmissionRisk."""
    return [
        ReadmissionRisk.config(
            binarize_days=[30, 90],
            train_mask=task_masks.Train.BASE,
            scale_pos_weight=1.0,
            eval_masks=[task_masks.Eval.BASE],
            loss_weight=1.0,
            discard_final_admission=False,
        )
    ]
