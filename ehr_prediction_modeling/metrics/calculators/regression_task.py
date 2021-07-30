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
"""Metric calculator for regression tasks."""

import math
from typing import List, Union

from ehr_prediction_modeling.metrics.calculators import base
import numpy as np


ListOrArray = Union[List[Union[float, int]], np.ndarray]


class RegressionTaskMetricCalculator(base.MetricCalculatorBase):
  """Calculator class for evaluation metrics for regression tasks."""

  ALL_METRICS = [
      "l1", "l1_percentage", "std", "nb_ex", "target_mean", "target_std"
  ]

  def __init__(self, preds: ListOrArray, targets: ListOrArray,
               weights: ListOrArray) -> None:
    """Initialize the auxiliary metrics calculator.

    Args:
      preds: The prediction values.
      targets: The target values.
      weights: The target weights.
    """
    self._preds = np.asarray(preds)
    self._targets = np.asarray(targets)
    self._abs_pred_target_diff = np.absolute(self._preds - self._targets)
    self._abs_percentage_pred_target_diff = np.absolute(
        (self._preds - self._targets) / self._targets)
    self._weights = weights

  def nb_ex(self) -> int:
    """Get the total number of examples over which to compute metrics."""
    return np.sum(self._weights)

  def l1(self) -> float:
    if np.sum(self._weights) > 0:
      return np.average(self._abs_pred_target_diff, weights=self._weights)
    return np.nan

  def l1_percentage(self) -> float:
    if np.sum(self._weights) > 0:
      return np.average(
          self._abs_percentage_pred_target_diff, weights=self._weights)
    return np.nan

  def std(self) -> float:
    if np.sum(self._weights) > 0:
      variance = np.average(
          self._abs_pred_target_diff**2, weights=self._weights)
      return math.sqrt(variance)
    return np.nan

  def target_mean(self) -> float:
    return np.average(self._targets)

  def target_std(self) -> float:
    return np.std(self._targets)
