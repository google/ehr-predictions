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
"""LOS task implementation to be used in experiments."""
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


class LengthOfStay(base_task.Task):
  """Task implementation for length of stay prediction (LOS).

  If a list of binarize_days are specified, this task predicts if the length of
  stay is <= n for each n in the list binarize_days.

  If binarize_days are not specified, this is a regression task to
  predict the exact length of stay.
  """

  task_type = types.TaskTypes.LOS

  @property
  def default_masks(self) -> List[str]:
    return [
        mask_utils.IGNORE_MASK, mask_utils.PADDED_EVENT_MASK,
        mask_utils.END_OF_ADMISSION_MASK
    ]

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
    if config.binarize_days:
      base_label = "length_of_stay_less_or_equal_%dd"
      self._target_names = [base_label % day for day in config.binarize_days]
    else:
      self._target_names = ["remaining_length_of_stay"]
    label_keys = self._target_names
    super().__init__(config, label_keys=label_keys)

  @property
  def target_names(self) -> List[str]:
    return self._target_names

  @property
  def num_targets(self) -> int:
    return len(self._target_names)

  @property
  def window_hours(self) -> List[int]:
    return [d * 24 for d in self._config.binarize_days or []]

  @property
  def prediction_task_type(self) -> str:
    if self._config.binarize_days:
      return types.TaskType.BINARY_CLASSIFICATION
    else:
      return types.TaskType.REGRESSION


  def get_label_dicts(
      self
  ) -> Tuple[Dict[Optional[str], Union[
      tf.FixedLenSequenceFeature, tf.FixedLenFeature]], Dict[Union[
          str, None], Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]]]:
    """Gets the feature dictionaries to parse a tf.SequenceExample.

    These dicts are used in the parse op (either batched or non-batched):
    https://www.tensorflow.org/api_docs/python/tf/io/parse_single_sequence_example#args
    They should include any label that is used by this task.

    Returns:
      context_d: Dictionaries of TF features to read in context.
      sequence_d: Dictionaries of TF features to read in sequence.
    """
    context_d, sequence_d = mask_utils.get_labels_for_masks(
        self._config.train_mask, self._config.eval_masks,
        self._all_supported_masks, self._config.time_since_event_label_key)

    sequence_d[label_utils.LOS_LABEL] = tf.FixedLenSequenceFeature(
        [1], tf.float32)

    return context_d, sequence_d

  def _binarized_los_labels(self, binarize_days: List[int],
                            batch: batches.TFBatch) -> tf.Tensor:
    """Gets the binarized LOS labels for a given batch.

    Args:
      binarize_days: list of integers, the days around which to binarize LOS.
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.

    Returns:
      Tensor in time-major shape wnct [num_unroll, batch_size,
      channels, num_targets], with value 1 if the remaining LOS is less than or
      equal to the given binarized day for that target, and 0 if the remaining
      LOS is greater than the binarized day for that target.
    """
    binarized_labels = []
    for day in binarize_days:
      binarized_batch_labels = tf.less_equal(
          batch.sequences[label_utils.LOS_LABEL],
          tf.constant(day, dtype=tf.float32))
      binarized_labels.append(binarized_batch_labels)
    binarized_labels_concat = tf.stack(binarized_labels, axis=3)
    return tf.cast(binarized_labels_concat, tf.float32)

  def _get_all_task_variables(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.TaskVariables:
    """Computes variables for LengthOfStay task.

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
        loss_type=self._config.loss_type,
        scale_pos_weight=self._config.get("scale_pos_weight", 1.0))
    eval_mask_dict = self.get_eval_mask_dict(batch)

    if self.prediction_task_type == types.TaskType.BINARY_CLASSIFICATION:
      predictions = tf.sigmoid(logits)
    else:
      predictions = logits

    return task_data.TaskVariables(
        loss=loss,
        targets=targets,
        predictions=predictions,
        train_mask=train_loss_mask,
        eval_mask_dict=eval_mask_dict,
    )

  def get_targets(
      self,
      batch: batches.TFBatch,
  ) -> tf.Tensor:
    if self._config.binarize_days:
      return self._binarized_los_labels(self._config.binarize_days, batch)
    else:
      return self._extract_labels(batch, [label_utils.LOS_LABEL])

  @classmethod
  def config(
      cls,
      eval_masks: List[str],
      train_mask: str,
      loss_type: str,
      time_buckets_per_day: int = 4,
      loss_weight: float = 1.0,
      binarize_days: Optional[List[int]] = None,
      accumulate_logits: bool = True,
      time_since_event_hours_list: Optional[List[int]] = None,
      scale_pos_weight: Optional[float] = 1.,
      task_layer_type: str = types.TaskLayerTypes.MLP,
      task_layer_sizes: Optional[List[int]] = None,
      regularization_type: str = types.RegularizationType.NONE,
      regularization_weight: float = 0.,
      name: str = "LengthOfStay",
      snr_config: Optional[configdict.ConfigDict] = None,
  ):
    """Generates a config object for LengthOfStay.

    Args:
      eval_masks: list of str, names of the masks to be used in eval. The names
        should be in LengthOfStay._supported_eval_masks.
      train_mask: str, name of the mask used for training. One of
        LengthOfStay._supported_train_masks.
      loss_type: str, type of loss to be used.
      time_buckets_per_day: Number of time buckets within a day. It is only
        needed for time since event masks.
      loss_weight: float, weight of this task loss.
      binarize_days: list of int, time periods for which to predict
        length_of_stay <= time window. Optional; if None, this is a regression
        task to predict the exact length of stay.
      accumulate_logits: bool, whether to create a CDF over the logits of
        increasing time_windows to encourage monotonicity.
      time_since_event_hours_list: If time since event mask is present in either
        train or eval, this is a list specifying the number of hours after
        event at which the model predicts LOS e.g. [0, 12] would predict LOS
        at event and 12 hours after event. If masking during training,
        these are combined i.e. the loss is comprised of the prediction at
        event and the prediction 12 hours after event. If masking during
        eval, these are returned as separate masks to get metrics for predicting
        length of stay at event, and separately for predicting length of
        stay at 12 hours after event.
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
      A ConfigDict to be used to instantiate a LOS task.
    """
    config = configdict.ConfigDict()
    config.task_type = LengthOfStay.task_type
    config.name = name
    config.binarize_days = (sorted(binarize_days) if binarize_days else None)
    if binarize_days and loss_type in [
        types.TaskLossType.L1, types.TaskLossType.L2
    ]:
      raise ValueError("Binary LengthOfStay task specified but L1/L2 loss "
                       "specified.")
    if not binarize_days and loss_type in [
        types.TaskLossType.CE, types.TaskLossType.BRIER
    ]:
      raise ValueError("Regression LengthOfStay task specified but CE/Brier "
                       "loss specified.")
    config.time_since_event_hours_list = time_since_event_hours_list or []
    # Since event label key used is time since admission.
    config.time_since_event_label_key = label_utils.TSA_LABEL
    config.time_buckets_per_day = time_buckets_per_day
    config.loss_type = loss_type
    config.eval_masks = eval_masks
    config.train_mask = train_mask
    config.accumulate_logits = accumulate_logits
    config.loss_weight = loss_weight
    config.scale_pos_weight = scale_pos_weight
    config.task_layer_sizes = task_layer_sizes or []
    config.regularization_type = regularization_type
    config.regularization_weight = regularization_weight
    return config

  @classmethod
  def default_configs(cls) -> List[configdict.ConfigDict]:
    """Generates a default config objects for binarized and regression LoS."""
    return [
        LengthOfStay.config(
            train_mask=task_masks.Train.BASE,
            eval_masks=[task_masks.Eval.BASE],
            loss_type=types.TaskLossType.CE,
            scale_pos_weight=1.0,
            loss_weight=1.0,
            binarize_days=[2, 7],
            name=types.TaskNames.BINARIZED_LOS),
        LengthOfStay.config(
            train_mask=task_masks.Train.BASE,
            eval_masks=[task_masks.Eval.BASE],
            scale_pos_weight=None,
            loss_type=types.TaskLossType.L2,
            loss_weight=1.0,
            name=types.TaskNames.REGRESSION_LOS)
    ]
