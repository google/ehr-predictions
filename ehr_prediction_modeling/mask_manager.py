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

"""Utilities for calculating masks.

Mask values are, 1 for values that are *kept* vs 0 for values that are *masked
out*.
"""
import copy
import functools
from typing import Any, Dict, List, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.utils import batches
from ehr_prediction_modeling.utils import label_utils
from ehr_prediction_modeling.utils import mask_utils
import tensorflow.compat.v1 as tf

from ehr_prediction_modeling import configdict


class MaskManager:
  """Mask manager."""

  def __init__(
      self,
      task_config: configdict.ConfigDict,
      window_hours: List[int],
      label_keys: List[str],
      supported_train_masks: Dict[str, List[str]],
      supported_eval_masks: Dict[str, List[str]],
  ):
    self._task_type = task_config.task_type

    if self._task_type == types.TaskTypes.READMISSION:
      self._discard_final_admission = task_config.discard_final_admission

    # List of hours since event to use for train mask.
    self._time_since_event_hours_list = task_config.get(
        "time_since_event_hours_list", [])
    # Label key for numerical time since event value. Default is time since
    # admission key.
    self._time_since_event_label_key = task_config.get(
        "time_since_event_label_key", label_utils.TSA_LABEL)
    self._time_buckets_per_day = task_config.get("time_buckets_per_day", None)
    self._window_hours = window_hours
    self._label_keys = label_keys
    self._supported_train_masks = supported_train_masks
    self._supported_eval_masks = supported_eval_masks
    self._all_supported_masks = copy.copy(self._supported_train_masks)
    self._all_supported_masks.update(self._supported_eval_masks)

    if self._task_type == types.TaskTypes.LAB_REGRESSION:
      # If there are multiple aggregations for each lab window_hours contains
      # duplicated values and its length is the real number of targets.
      self._num_targets = len(window_hours)
    else:
      self._num_targets = len(self._label_keys)



  def _get_component_mask(
      self,
      composite_mask_name: str,
      component_mask_name: str,
      batch: batches.TFBatch,
  ) -> Union[Dict[Union[str, int], tf.Tensor], tf.Tensor]:
    """Gets a mask from a batch of data.

    Args:
      composite_mask_name: str, the name of the composite mask.
      component_mask_name: str, the name of the mask to be extracted.
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.

    Returns:
      Tensor in time-major shape wnct
      [num_unroll, batch_size, channels, num_targets] with the mask value.
    """
    if (component_mask_name
        not in self._all_supported_masks[composite_mask_name]):
      raise ValueError(
          f"Component {component_mask_name} is not part of composite "
          f"{composite_mask_name}, its components are "
          f"{self._all_supported_masks[composite_mask_name]}")

    mask_name_to_fn = {
        mask_utils.SINCE_EVENT_TRAIN_MASK:
            self._time_since_event_train_mask,
        mask_utils.SINCE_EVENT_EVAL_MASK:
            functools.partial(
                self._time_since_event_eval_mask, composite_mask_name),
        mask_utils.AT_DISCHARGE_MASK:
            self._at_discharge_mask,
        mask_utils.INTERVAL_MASK:
            self._interval_mask,
        mask_utils.PATIENT_MASK:
            self._patient_mask,
        mask_utils.END_OF_ADMISSION_MASK:
            self._end_of_admission_mask,
        mask_utils.IGNORE_MASK:
            self._ignore_mask,
        mask_utils.PADDED_EVENT_MASK:
            self._padded_event_mask,
        mask_utils.UNKNOWN_LOOKAHEAD_MASK:
            self._unknown_lookahead_mask,

    }

    if component_mask_name in mask_name_to_fn:
      return mask_name_to_fn[component_mask_name](batch)
    else:
      raise ValueError(f"Unknown component mask {component_mask_name}")

  def get_masks(
      self,
      mask_names: List[str],
      batch: batches.TFBatch,
  ) -> Dict[str, tf.Tensor]:
    """Gets a dict of masks for a given batch.

    Args:
      mask_names: list of str, corresponding to composite masks in
        _supported_train_masks or _supported_eval_masks.
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.

    Returns:
      Dict of str, the mask name, to mask tensor in time-major shape wnct
      [num_unroll, batch_size, channels, num_targets]. The value of the mask
      mask tensor is the cross-multiplication of the component boolean masks as
      specified in _supported_train_masks or _supported_eval_masks.
    """
    mask_dict = {}
    for mask_name in mask_names:
      mask_dict[mask_name] = self._build_composite_mask_from_components(
          composite_mask_name=mask_name,
          batch=batch,
      )
    return mask_dict

  def _build_composite_mask_from_components(self, composite_mask_name: str,
                                            batch: batches.TFBatch) -> None:
    """Returns a composite mask.

    The SINCE_EVENT_MASK has a different behavior depending on model
    mode. During training, its different components (one per
    `_time_since_event_hours_list`) are summed. During eval, they are kept
    separate so that it is possible to evaluate at each chosen time after
    event.

    Args:
      composite_mask_name: str. Name of the mask.
      batch: container for a batch of data

    Returns:
      A mask tensor in time-major shape wnct
      [num_unroll, batch_size, channels, num_targets]. The value of the mask
      mask tensor is the cross-multiplication of the component boolean masks as
      specified in _supported_train_masks or _supported_eval_masks for the
      composite_mask_name.
    """
    computed_masks = []
    for component_mask_name in self._all_supported_masks[composite_mask_name]:
      computed_masks.append(
          self._get_component_mask(
              composite_mask_name=composite_mask_name,
              component_mask_name=component_mask_name,
              batch=batch))
    return mask_utils.get_combined_mask(computed_masks)

  def _time_since_event_eval_mask(
      self, composite_mask_name: str, batch: batches.TFBatch
      ) -> tf.Tensor:
    """Returns mask generated for time since event eval.

    Args:
      composite_mask_name: Name of the composite mask this mask is being
        generated for. Used to parse the hours after event that should be
        set for the mask.
      batch: Batch of the data to create a mask for.

    Returns:
      A single mask with a positive value for any event at hours
      since event parsed from the composite mask name.
    """
    hours = mask_utils.get_time_since_event_mask_hours(composite_mask_name)
    return self._time_since_event_mask(batch, target_hours=hours)

  def _time_since_event_train_mask(
      self, batch: batches.TFBatch) -> tf.Tensor:
    """Returns mask generated for specific times since some event of interest.

    Mask generated will have a positive value for any hour in
    self._time_since_event_hours_list that is also positive in the base_mask.

    Args:
      batch: Batch of the data to create a mask for.

    Returns:
      A single mask with a positive value for any event at hours
      after some event for every hour in self._time_since_event_hours_list.
    """
    if not self._time_since_event_hours_list:
      raise ValueError("Time since masks require a non-empty list of "
                       "times that indicate how many hours after the event"
                       " the mask will be nonzero.")
    time_mask = self._time_since_event_mask(
        batch=batch, target_hours=self._time_since_event_hours_list[0])
    for hours in self._time_since_event_hours_list[1:]:
      time_mask += self._time_since_event_mask(batch, hours)
    return time_mask

  def _time_since_event_mask(
      self, batch: batches.TFBatch,
      target_hours: int) -> tf.Tensor:
    """Mask that is 1 for events at hours since some event of interest.

    To accommodate shifting between time since event timestamps and the
    target hours, we take the difference as (time since event - target time)
    and select only those timesteps which have a difference >=0 (happen at
    or after the target) and < bucket length (closest time after target).

    Args:
      batch: Batch of the data to create a mask for.
      target_hours: target number of hours since event to set mask to 1.

    Returns:
      A single mask with a positive value for any event at target hours
      after some event.
    """
    if not self._time_buckets_per_day:
      raise ValueError(
          "Must specify time_buckets_per_day in the task config if using a "
          "time since event mask.")
    time_since_event = self._extract_mask_from_sequence(
        batch, self._time_since_event_label_key)
    target_sortable_time = tf.constant(
        target_hours / 24., dtype=tf.float32)
    diff_since_target = time_since_event - target_sortable_time
    after_target_time = tf.cast(
        tf.greater_equal(
            diff_since_target, tf.constant(
                0., dtype=tf.float32)),
        tf.float32)
    # We subtract 1e-6 from the upper bound to account for the case in which
    # there is a trigger time exactly one bucket away from the target time. Due
    # to precision issues when the length of a bucket has many decimal places,
    # this timestamp can be slightly smaller than the exact length of the
    # bucket, resulting in 2 eligible trigger times instead of 1.
    less_than_one_bucket_from_target_time = tf.cast(
        tf.less(
            diff_since_target, tf.constant(
                (1. / self._time_buckets_per_day) - 1e-6,
                dtype=tf.float32)),
        tf.float32)
    return after_target_time * less_than_one_bucket_from_target_time

  def _at_discharge_mask(self,
                         batch: batches.TFBatch) -> tf.Tensor:
    """Returns 1 for discharge events and 0 for the rest."""
    per_target_list = [batch.sequences[mask_utils.AT_DISCHARGE_MASK]
                      ] * self._num_targets
    # Note that the label 2 represents the final event of an admission that is
    # also the final episode of a given medical history for which the patient
    # was discharged.
    if self._discard_final_admission:
      # In that case only labels 1 are considered.
      return tf.cast(
          tf.equal(
              tf.stack(per_target_list, axis=3), 1),
          tf.float32)
    else:
      # In that case both labels 1 and 2 are considered.
      return tf.cast(
          tf.greater_equal(
              tf.stack(per_target_list, axis=3), 1),
          tf.float32)

  def _interval_mask(self, batch: batches.TFBatch) -> tf.Tensor:
    return self._extract_mask_from_sequence(
        batch,
        label_utils.SEGMENT_LABEL,
        invert=True,
    )

  def _patient_mask(self, batch: batches.TFBatch) -> tf.Tensor:
    return self._extract_mask_from_context(
        batch,
        label_utils.CENSORED_PATIENT_LABEL,
        invert=True,
    )

  def _end_of_admission_mask(self,
                             batch: batches.TFBatch) -> tf.Tensor:
    """Returns 0 for the final event of an admission, and 1 otherwise."""
    los_labels = self._extract_mask_from_sequence(batch, label_utils.LOS_LABEL)
    return tf.cast(
        tf.not_equal(
            los_labels,
            tf.constant(0, dtype=tf.float32)),
        tf.float32)

  def _extract_labels(self, batch: batches.TFBatch) -> tf.Tensor:
    """Extracts the labels denoted by label_keys from the data.

    Args:
      batch: containing a batch of data.

    Returns:
      Tensor in time-major shape wnct
      [num_unroll, batch_size, channels, num_targets] with the labels for each
      key given in label_keys.
    """
    return tf.stack(
        [batch.sequences[label_key] for label_key in self._label_keys], axis=3)

  def _extract_mask_from_context(self,
                                 batch: batches.TFBatch,
                                 mask_label_name: str,
                                 invert: bool = False) -> tf.Tensor:
    """Extracts a mask from the batch context.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      mask_label_name: string, the name of the label required.
      invert: bool. whether to invert the mask before returning.

    Returns:
      Tensor in time-major shape wnct [num_unroll, batch_size, channels,
      num_targets] with the float mask value.
    """
    per_target_list = self._num_targets * [
        tf.expand_dims(batch.context[mask_label_name], -1)
    ]
    mask_concat = tf.stack(per_target_list, axis=3)
    mask_concat = tf.cast(mask_concat,
                                          tf.float32)
    if invert:
      mask_concat = tf.ones_like(
          mask_concat, dtype=tf.float32) - mask_concat
    return tf.cast(
        mask_concat, dtype=tf.float32)

  def _extract_mask_from_sequence(self,
                                  batch: batches.TFBatch,
                                  mask_label_name: str,
                                  invert: bool = False) -> tf.Tensor:
    """Extracts a mask from the sequences in a batch.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      mask_label_name: string, the name of the label required.
      invert: bool, whether to invert the mask before returning.

    Returns:
      Tensor in time-major shape wnct [num_unroll, batch_size, channels,
      num_targets] with the float mask value.
    """
    per_target_list = [batch.sequences[mask_label_name]] * self._num_targets
    mask_concat = tf.stack(per_target_list, axis=3)
    mask_concat = tf.cast(mask_concat,
                                          tf.float32)
    if invert:
      mask_concat = tf.ones_like(
          mask_concat, dtype=tf.float32) - mask_concat
    return tf.cast(
        mask_concat, dtype=tf.float32)

  def _ignore_mask(self, batch: batches.TFBatch) -> tf.Tensor:
    """Mask that is set to 0 for outpatient and unknown TOD events."""
    return self._extract_mask_from_sequence(
        batch, label_utils.IGNORE_LABEL, invert=True)

  def _padded_event_mask(self,
                         batch: batches.TFBatch) -> tf.Tensor:
    """Mask that is 0 for timestamps less than or equal to 0 i.e. padded."""
    timestamps = self._extract_mask_from_sequence(batch,
                                                  label_utils.TIMESTAMP_KEY)
    return tf.cast(
        tf.greater(
            timestamps,
            tf.constant(0, dtype=tf.float32)),
        tf.float32)

  def _unknown_lookahead_mask(self,
                              batch: batches.TFBatch) -> tf.Tensor:
    """Extracts a mask to ignore unknown lookaheads from the data."""
    if self._task_type == types.TaskTypes.LAB_REGRESSION:
      targets = self._extract_labels(batch)
      is_positive = tf.greater(
          targets, tf.zeros_like(targets))
      return tf.cast(
          is_positive, dtype=tf.float32)
    else:
     raise ValueError("UNKNOWN_LOOKAHEAD_MASK is defined only for tasks "
                      f"{types.TaskTypes.LAB_REGRESSION} but the task is "
                      f"{self._task_type}")

