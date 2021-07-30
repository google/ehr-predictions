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
"""PR and ROC curves and calculations."""
from typing import List, Optional, Union
import numpy as np

ListOrArray = Union[List[Union[float, int]], np.ndarray]
CurvePoints = Union[List[List[float]], np.ndarray]


def area_under_curve_from_raw_scores(scores_pos: ListOrArray,
                                     scores_neg: ListOrArray) -> float:
  """Calculates the AUC from the sequences of positive and negative scores.

  Args:
    scores_pos: The scores of the examples of positive class.
    scores_neg: The scores of the examples of negative class.

  Returns:
    The area under the ROC curve.
  """
  # ROC curve is already sorted, ascending according to the x-values (first item
  # in the pairs).
  roc_curve = roc_curve_from_raw_scores(scores_pos, scores_neg)
  return _curve_area(roc_curve)


def pr_area_under_curve_from_raw_scores(scores_pos: ListOrArray,
                                        scores_neg: ListOrArray) -> float:
  """Calculates the PRAUC from the sequences of positive and negative scores.

  Args:
    scores_pos: The scores of the examples of positive class.
    scores_neg: The scores of the examples of negative class.

  Returns:
    The area under the PR curve.
  """
  # PR curve is already sorted, ascending according to the x-values (first item
  # in the pairs).
  pr_curve = pr_curve_from_raw_scores(scores_pos, scores_neg)
  return _curve_area(pr_curve)


def weighted_area_under_curve_from_raw_scores(
    scores_pos: ListOrArray, scores_neg: ListOrArray,
    weights_pos: ListOrArray, weights_neg: ListOrArray) -> float:
  """Calculates the AUC from the sequences of positive and negative scores.

  Args:
    scores_pos: The scores of the examples of positive class.
    scores_neg: The scores of the examples of negative class.
    weights_pos: The weights of the examples of positive class.
    weights_neg: The weights of the examples of negative class.

  Returns:
    The area under the ROC curve.
  """
  # ROC curve is already sorted, ascending according to the x-values (first item
  # in the pairs).
  roc_curve = roc_curve_from_raw_scores(scores_pos, scores_neg,
                                        weights_pos, weights_neg)
  return _curve_area(roc_curve)


def weighted_pr_area_under_curve_from_raw_scores(
    scores_pos: ListOrArray, scores_neg: ListOrArray,
    weights_pos: ListOrArray, weights_neg: ListOrArray) -> float:
  """Calculates the PRAUC from the sequences of positive and negative scores.

  Args:
    scores_pos: The scores of the examples of positive class.
    scores_neg: The scores of the examples of negative class.
    weights_pos: The weights of the examples of positive class.
    weights_neg: The weights of the examples of negative class.

  Returns:
    The area under the PR curve.
  """
  # PR curve is already sorted, ascending according to the x-values (first item
  # in the pairs).
  pr_curve = pr_curve_from_raw_scores(scores_pos, scores_neg,
                                      weights_pos, weights_neg)
  return _curve_area(pr_curve)


def roc_curve_from_raw_scores(
    scores_pos: ListOrArray,
    scores_neg: ListOrArray,
    weights_pos: Optional[ListOrArray] = None,
    weights_neg: Optional[ListOrArray] = None) -> CurvePoints:
  """Computes the ROC curve from the sequences of positive and negative scores.

  This is based on the 'Algorithm 1' described in 'An introduction to ROC
  analysis' by Tom Fawcett, available at:
  http://people.inf.elte.hu/kiss/11dwhdm/roc.pdf
  The instance important weights are optional, but can be used to influence the
  computation of TP and FP rate and the ROC points, in order to compute
  weighted AUC scores.

  Args:
    scores_pos: The scores of the examples of positive class.
    scores_neg: The scores of the examples of negative class.
    weights_pos: The weights of the examples of positive class.
    weights_neg: The weights of the examples of negative class.

  Returns:
    A list of points representing the ROC curve.

  Raises:
    ValueError: If any of the scores lists are given as None.
    ValueError: If all weights are 0, or at least one is negative.
  """
  if scores_pos is None or scores_neg is None:
    raise ValueError("None value given as input.")
  if weights_pos is None:
    weights_pos = np.ones_like(scores_pos)
  if weights_neg is None:
    weights_neg = np.ones_like(scores_neg)
  if np.asarray(weights_pos).size > 0 and np.nanmin(weights_pos) < 0.:
    raise ValueError("Weights should be positive (pos)")
  if np.asarray(weights_neg).size > 0 and np.nanmin(weights_neg) < 0.:
    raise ValueError("Weights should be positive (neg)")
  num_positives = np.sum(weights_pos)
  num_negatives = np.sum(weights_neg)
  # For simplicity, first we merge the lists together while keeping track of the
  # positive/negative labels.
  all_scores = np.append(scores_pos, scores_neg)
  all_weights = np.append(weights_pos, weights_neg)
  is_positive_class = np.zeros_like(all_scores, dtype=bool)
  is_positive_class[:len(scores_pos)] = True
  sorting_perm = all_scores.argsort()
  is_positive_class_sorted = is_positive_class[sorting_perm][::-1]
  all_scores_sorted = all_scores[sorting_perm][::-1]
  all_weights_sorted = all_weights[sorting_perm][::-1]
  false_positives = 0
  true_positives = 0
  roc_points = []
  previous_score = np.NINF
  for score, is_positive, weight in zip(
      all_scores_sorted, is_positive_class_sorted, all_weights_sorted):
    if not np.isclose(previous_score, score):
      roc_points.append([false_positives / num_negatives,
                         true_positives / num_positives])
      previous_score = score
    if is_positive:
      true_positives += weight
    else:
      false_positives += weight
  roc_points.append([1., 1.])
  return roc_points


def pr_curve_from_raw_scores(
    scores_pos: ListOrArray,
    scores_neg: ListOrArray,
    weights_pos: Optional[ListOrArray] = None,
    weights_neg: Optional[ListOrArray] = None) -> CurvePoints:
  """Computes the PR curve from the sequences of positive and negative scores.

  This is based on the 'Algorithm 1' described in 'An introduction to ROC
  analysis' by Tom Fawcett, available at:
  http://people.inf.elte.hu/kiss/11dwhdm/roc.pdf
  The instance important weights are optional, but can be used to influence the
  computation of TP and FP rate and the PR points, in order to compute
  weighted AUC scores.

  Args:
    scores_pos: The scores of the examples of positive class.
    scores_neg: The scores of the examples of negative class.
    weights_pos: The weights of the examples of positive class.
    weights_neg: The weights of the examples of negative class.

  Returns:
    A list of points representing the PR curve.

  Raises:
    ValueError: If any of the scores lists are given as None.
    ValueError: If all weights are 0, or at least one is negative.
  """
  if scores_pos is None or scores_neg is None:
    raise ValueError("None value given as input.")
  if weights_pos is None:
    weights_pos = np.ones_like(scores_pos)
  if weights_neg is None:
    weights_neg = np.ones_like(scores_neg)
  if np.asarray(weights_pos).size > 0 and np.nanmin(weights_pos) < 0.:
    raise ValueError("Weights should be positive (pos)")
  if np.asarray(weights_neg).size > 0 and np.nanmin(weights_neg) < 0.:
    raise ValueError("Weights should be positive (neg)")
  num_positives = np.sum(weights_pos)
  # For simplicity, first we merge the lists together while keeping track of the
  # positive/negative labels.
  all_scores = np.append(scores_pos, scores_neg)
  all_weights = np.append(weights_pos, weights_neg)
  is_positive_class = np.zeros_like(all_scores, dtype=bool)
  is_positive_class[:len(scores_pos)] = True
  sorting_perm = all_scores.argsort()
  is_positive_class_sorted = is_positive_class[sorting_perm][::-1]
  all_scores_sorted = all_scores[sorting_perm][::-1]
  all_weights_sorted = all_weights[sorting_perm][::-1]
  false_positives = 0
  true_positives = 0
  pr_points = []
  previous_score = all_scores_sorted[0] if all_scores_sorted.size else np.NINF
  for score, is_positive, weight in zip(
      all_scores_sorted, is_positive_class_sorted, all_weights_sorted):
    if not np.isclose(previous_score, score):
      # We ignore examples with weights 0.
      if true_positives + false_positives != 0.:
        if not pr_points:
          # We need to report the first point on the y axis
          pr_points.append(
              [0., true_positives / (true_positives + false_positives)])
        pr_points.append([true_positives / num_positives,
                          true_positives / (true_positives + false_positives)])
      previous_score = score
    if is_positive:
      true_positives += weight
    else:
      false_positives += weight
  if true_positives + false_positives != 0.:
    if not pr_points:
      # We need to report the first point on the y axis
      pr_points.append(
          [0., true_positives / (true_positives + false_positives)])
    pr_points.append([1., true_positives / (true_positives + false_positives)])
  return pr_points


def _curve_area(curve: CurvePoints) -> float:
  """Computes the area under a curve.

  Args:
    curve: [[x, y], ...] List of points of the curve (ordered by x axis).

  Returns:
    The area of the curve.
  """
  area = 0
  for left, right in zip(curve, curve[1:]):
    area += _trapezoid_area(left, right)
  return area


def _trapezoid_area(point_left: List[float], point_right: List[float]) -> float:
  """Computes the area under the line between the given control points.

  Args:
    point_left: ([x, y]) The upper left corner of the trapesoid.
    point_right: ([x, y]) The upper right corner of the trapesoid.

  Returns:
    The area of the trapesoid.
  """
  left_x, left_y = point_left
  right_x, right_y = point_right
  base = abs(right_x - left_x)
  height = (left_y + right_y) / 2.
  return base * height
