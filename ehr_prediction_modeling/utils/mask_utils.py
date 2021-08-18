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
"""Defining some masks' names and utils."""
import re
from typing import Dict, List, Tuple, Union

from ehr_prediction_modeling.utils import label_utils
import tensorflow.compat.v1 as tf

# Names of component masks available to tasks.
# Masks intervals for patient history based on labels set for censoring during
# preprocessing.
INTERVAL_MASK = "interval_mask"
# Masks entire patients based on labels set for censoring during preprocessing.
PATIENT_MASK = "patient_mask"

UNKNOWN_LOOKAHEAD_MASK = "unknown_lookahead_mask"
IGNORE_MASK = "ignore_mask"
PADDED_EVENT_MASK = "padded_event_mask"
SINCE_EVENT_TRAIN_MASK = "since_event_train_mask"
SINCE_EVENT_EVAL_MASK = "since_event_eval_mask"
AT_DISCHARGE_MASK = label_utils.DISCHARGE_LABEL
END_OF_ADMISSION_MASK = "end_of_admission_mask"
SINCE_EVENT_MASK_SUFFIX = "hours_since_event"

TFFeatureTypes = Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]
LabelDicts = Tuple[Dict[str, TFFeatureTypes], Dict[str, TFFeatureTypes]]



def get_labels_for_masks(
    train_mask: str,
    eval_masks: List[str],
    all_supported_masks: Dict[str, List[str]],
    time_since_event_label_key=label_utils.TSA_LABEL,
) -> LabelDicts:
  """Gets label dictionary required for masks from the context and sequence.

  Args:
    train_mask: Train masks to fetch labels for.
    eval_masks: Eval masks to fetch labels for.
    all_supported_masks: Mapping of composite masks to components for the task.
    time_since_event_label_key: The numerical label key which measures the time
      since the relevant event. By default this key is time since admission.
    min_aki_level: int, Only used for AKI task.

  Returns:
    context_d: Dictionaries of TF features to read in context.
    sequence_d: Dictionaries of TF features to read in sequence.
  """
  context_d = {}
  sequence_d = {}
  required_masks = []


  for mask_name in eval_masks + [train_mask]:
    required_masks.extend(all_supported_masks[mask_name])

  sequence_int_feature_labels = {
      INTERVAL_MASK:
          label_utils.SEGMENT_LABEL,
      IGNORE_MASK:
          label_utils.IGNORE_LABEL,
      AT_DISCHARGE_MASK:
          label_utils.DISCHARGE_LABEL,
  }

  for mask in set(required_masks):

    if mask in sequence_int_feature_labels.keys():
      sequence_d[
          sequence_int_feature_labels[mask]] = tf.FixedLenSequenceFeature(
              [1], tf.int64)
    elif mask in [SINCE_EVENT_TRAIN_MASK, SINCE_EVENT_EVAL_MASK]:
      sequence_d[time_since_event_label_key] = (
          tf.FixedLenSequenceFeature([1], tf.float32))
    elif mask == PATIENT_MASK:
      context_d[label_utils.CENSORED_PATIENT_LABEL] = tf.FixedLenFeature(
          [], tf.int64)
    elif mask in [
        UNKNOWN_LOOKAHEAD_MASK,
        PADDED_EVENT_MASK,
        END_OF_ADMISSION_MASK,
    ]:
      continue
    else:
      raise ValueError("No labels specified for %s" % mask)
  return context_d, sequence_d


def get_unique_mask_name(task_type: str, mask_name: str) -> str:
  """Prepends the task name to the mask name to ensure uniqueness.

  Args:
    task_type: Name of the task for which the masks are being modified.
    mask_name: Name of the mask e.g. base_eval.

  Returns:
    A mask name that has the task name prepended. Example: los_base_eval.
  """
  unique_mask_name = mask_name
  if task_type not in mask_name:
    unique_mask_name = "_".join([task_type, mask_name])
  return unique_mask_name


def get_combined_mask(component_masks: List[tf.Tensor]) -> tf.Tensor:
  """Returns the multiplication of a list of component masks.

  Args:
    component_masks: list of the component masks. Each component mask is a
      tensor in time-major shape wnct [num_unroll, batch_size, channels,
      num_targets] with the mask value (a boolean).

  Returns:
    The multiplication of each of the boolean component masks: a tensor in
      time-major shape wnct [num_unroll, batch_size, channels, num_targets].
  """
  base_mask = None
  for mask in component_masks:
    if base_mask is not None:
      if len(base_mask.shape) != len(mask.shape):
        raise ValueError("Expected masks of the same rank when combining, "
                         f"found {base_mask.shape} and {mask.shape}")
      base_mask *= mask
    else:
      base_mask = mask
  return base_mask


def get_time_since_event_mask_hours(mask_name: str) -> int:
  """Returns the hours from time since event mask name."""
  match = re.match(r".*_([\d]+)_%s.*" % SINCE_EVENT_MASK_SUFFIX, mask_name)
  if not match:
    raise ValueError("Malformed time since event mask name: %s" % mask_name)
  return int(match.group(1))


