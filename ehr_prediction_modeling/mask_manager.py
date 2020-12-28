# coding=utf-8
# Copyright 2020 Google Health Research.
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

    self._hours_after_admission = task_config.get("hours_after_admission", [])
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
    if (component_mask_name not in
        self._all_supported_masks[composite_mask_name]):
      raise ValueError(
          f"Component {component_mask_name} is not part of composite "
          f"{composite_mask_name}, its components are "
          f"{self._all_supported_masks[composite_mask_name]}")

    mask_name_to_fn = {
        mask_utils.AROUND_ADMISSION_TRAIN_MASK:
            self._adm_train_mask,
        mask_utils.AROUND_ADMISSION_EVAL_MASK:
            functools.partial(self._adm_eval_mask, composite_mask_name),
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

    The AROUND_ADMISSION_MASK has a different behavior depending on model
    mode. During training, its different components (one per
    `hours_after_admission`) are summed. During eval, they are kept separate
    so that it is possible to evaluate at each chosen time after admission.

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

  def _adm_eval_mask(self, composite_mask_name: str,
                     batch: batches.TFBatch) -> tf.Tensor:
    """Returns mask generated for around admission eval.

    Args:
      composite_mask_name: Name of the composite mask this around admission
        component mask is being generated for. Used to parse the hours after
        admission that should be set for the mask.
      batch: Batch of the data to create a mask for.

    Returns:
      A single mask with a positive value for any event at hours
      after admission parsed from the composite mask name.
    """
    hours = mask_utils.get_time_from_adm_mask(composite_mask_name)
    return self._around_admission_mask(batch, hours)

  def _adm_train_mask(self, batch: batches.TFBatch) -> tf.Tensor:
    """Returns mask generated for around admission training.

    Mask generated will have a positive value for any hour in
    self._hours_after_admission that is also positive in the base_mask.

    Args:
      batch: Batch of the data to create a mask for.

    Returns:
      A single mask with a positive value for any event at hours
      after admission for every hour in hours_list.
    """
    if not self._hours_after_admission:
      raise ValueError("Around admission masks require a non-empty list of "
                       "times that indicate how many hours after the admission"
                       " the mask will be nonzero.")
    time_mask = self._around_admission_mask(
        batch, self._hours_after_admission[0])
    for time in self._hours_after_admission[1:]:
      time_mask += self._around_admission_mask(batch, time)
    return time_mask

  def _around_admission_mask(self, batch: batches.TFBatch,
                             hours_after_admission: int) -> tf.Tensor:
    """Mask that is 1 for events exactly hours_after_admission since admission."""
    time_after_admission = self._extract_mask_from_sequence(
        batch, label_utils.TSA_LABEL)
    float_day = hours_after_admission / 24.
    return tf.cast(
        tf.equal(
            time_after_admission,
            tf.constant(
                float_day, dtype=tf.float32)),
        tf.float32)

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

