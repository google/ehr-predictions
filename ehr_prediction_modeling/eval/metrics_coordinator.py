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
"""Coordinator for metrics of various targets in evaluation."""

import collections
import os
from typing import List, Mapping, Optional, Tuple, Type

from absl import logging
import dataclasses
from ehr_prediction_modeling.eval import metrics_data
import numpy as np

import tensorflow.compat.v1 as tf


@dataclasses.dataclass(frozen=True)
class MetricsTarget(object):
  """Describes a target metrics will be computed for."""
  target_name: str
  split_name: str
  mask_name: Optional[str] = None


  def to_string(self) -> str:
    """Returns a string description of this target."""
    return (f"Target: {self.target_name}, Split: {self.split_name}, Mask: "
            f"{self.mask_name}")


class MetricsCoordinator(object):
  """Base class for accumulating data used to calculate metrics."""

  def __init__(self):
    """Initializes metrics coordinator."""
    self._data_dict = {}

  def add_data(self,
               metrics_data_type: Type[metrics_data.MetricsTypes],
               data: Tuple[float],
               target_name: str,
               split_name: str,
               mask_name: Optional[str] = None):
    """Extends object's lists of data.

    Args:
      metrics_data_type: The subclass of MetricsData that will be used to store
        this data.
      data: A batch of data to add. Tuple should contain data listed in the
        order required by the relevant MetricsData add_data method. For example,
        For RegressionMetricsData, data should be (predictions, targets,
        weights).
      target_name: Name of the target the data descibes.
      split_name: Name of the split the data is from.
      mask_name: Name of the mask used with the data given.

    Raises:
      ValueError if the data given is found invalid by the MetricsData class.
    """
    metrics_target = MetricsTarget(target_name, split_name, mask_name)
    if metrics_target not in self._data_dict:
      self._data_dict[metrics_target] = metrics_data_type()
    self._data_dict[metrics_target].add_data(*data)

  def _clear_data(self):
    """Clears any data in _data_dict."""
    self._data_dict = {}


  def log_metrics(self, current_step: int, clear_data: bool = False) -> None:
    """Logs all metrics, then may clear stored metrics data.

    Args:
      current_step: Current step we are evaluating.
      clear_data: If true, all stored data will be cleared.
    """
    for metrics_target, data in self._data_dict.items():
      data.log_metrics(metrics_target.to_string(), current_step)
    if clear_data:
      # Clear any data after it is logged.
      self._clear_data()


def add_batch_to_binary_metrics_data(metrics_coordinator, target_names,
                                     predictions, binary_targets,
                                     eval_mask_dict, split_name):
  """Adds binary tasks predictions to BinaryMetricsData objects.

  Args:
    metrics_coordinator: MetricsCoordinator, used to accumulate metrics for each
      target.
    target_names: list of str, the names of the targets in the task.
    predictions: array of predictions in time-major shape wnct [num_unroll,
      batch_size, channels, num_targets].
    binary_targets: array of binary targets in time-major shape wnct [
      num_unroll, batch_size, channels, num_targets].
    eval_mask_dict: dict of string mask name to array, the loss masks to be used
      in evaluation, in time-major shape wnct [num_unroll, batch_size, channels,
      num_targets].
    split_name: str, name of the data split this batch is from.
  """
  num_targets = len(target_names)
  # Split predictions by target into a list of length num_targets.
  predictions_per_target = np.split(
      predictions, indices_or_sections=num_targets, axis=3)

  for mask_name, mask in eval_mask_dict.items():
    positive_filter_and_mask = (mask * binary_targets).astype(bool)
    negative_filter_and_mask = (
        mask * (np.ones_like(binary_targets) - binary_targets)).astype(bool)

    positive_masks_per_target = np.split(
        positive_filter_and_mask, indices_or_sections=num_targets, axis=3)
    negative_masks_per_target = np.split(
        negative_filter_and_mask, indices_or_sections=num_targets, axis=3)

    for idx, target_name in enumerate(target_names):
      positives = predictions_per_target[idx][positive_masks_per_target[idx]]
      positive_weights = np.ones_like(positives)
      negatives = predictions_per_target[idx][negative_masks_per_target[idx]]
      negative_weights = np.ones_like(negatives)
      metrics_coordinator.add_data(
          metrics_data.BinaryMetricsData,
          (positives, negatives, positive_weights, negative_weights),
          target_name,
          split_name,
          mask_name=mask_name)


def add_batch_to_regression_metrics_data(metrics_coordinator, target_names,
                                         predictions, targets, eval_mask_dict,
                                         split_name):
  """Adds regression tasks predictions to RegressionMetricsData objects.

  Args:
    metrics_coordinator: MetricsCoordinator, used to accumulate metrics for each
      target.
    target_names: list of str, the names of the targets in the task.
    predictions: array of predictions in time-major shape wnct [num_unroll,
      batch_size, channels, num_targets].
    targets: array of float targets in time-major shape wnct [ num_unroll,
      batch_size, channels, num_targets].
    eval_mask_dict: dict of string mask name to array, the loss masks to be used
      in evaluation, in time-major shape wnct [num_unroll, batch_size, channels,
      num_targets].
    split_name: str, name of the data split this batch is from.
  """
  num_targets = len(target_names)
  predictions_per_target = np.split(
      predictions, indices_or_sections=num_targets, axis=3)
  target_list = np.split(
      targets, indices_or_sections=num_targets, axis=3)
  for mask_name, mask in eval_mask_dict.items():
    masks_per_target = np.split(
        mask, indices_or_sections=num_targets, axis=3)
    for idx, target_name in enumerate(target_names):
      predictions = predictions_per_target[idx][masks_per_target[idx].astype(
          bool)]
      targets = target_list[idx][masks_per_target[idx].astype(bool)]
      weights = np.ones_like(predictions)
      metrics_coordinator.add_data(
          metrics_data.RegressionMetricsData, (predictions, targets, weights),
          target_name,
          split_name,
          mask_name=mask_name)
