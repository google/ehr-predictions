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
"""Implements methods for measuring and resolving model calibration."""
from typing import List, Optional, Tuple, Union
import numpy as np


ListOrArray = Union[List[Union[float, int]], np.ndarray]

# Our data is quite large, so there will be plenty of predictions. No need to
# use a small number of bins.
DEFAULT_NUM_BINS = 50


class Error(Exception):
  pass


class MismatchedPredictionArraysError(Error):
  """Raised when the passed arrays should have the same size, but don't."""


def reliability_histogram(
    predictions: ListOrArray,
    labels: ListOrArray,
    confidence_scores: ListOrArray,
    num_bins: int = DEFAULT_NUM_BINS,
    example_weights: Optional[ListOrArray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes a histogram of confidence vs accuracy for the specified # of bins.

  In the resulting histogram, the confidence is the x axis (buckets) and the
  bin values correspond to the accuracy within each bucket. The reliability
  diagram/histogram is often used and has been discussed for instance in
  (DeGroot & Fienberg, 1983), (Niculescu-Mizil & Caruana, 2005),
  (Guo et al., 2017).

  Args:
    predictions: Non-soft predictions made by the model. In the binary
      classification case this would mean an integer array of 0-s and
      1-s. Worth noting, since in our models we sometimes refer to predictions
      as a soft array of class probabilities.
    labels: Original data labels.
    confidence_scores: Confidence scores produced by the model.
    num_bins: The number of bins to use when computing the diagram.
    example_weights: If specified, the histogram will be computed from the
      weighted averages instead.

  Returns:
    (accuracy_histogram, confidence_histogram, counts) where:
      accuracy_histogram is the accuracies corresponding to each
          bin amongst bins that reflect confidence ranges from i/M to (i+1)/M.
      confidence_histogram is the confidences corresponding to each
          bin amongst bins that reflect confidence ranges from i/M to (i+1)/M.
      counts gives the number of examples per each bin.
  """
  bin_example_counts = np.zeros(num_bins, dtype=np.float32)
  bin_accurate_predictions = np.zeros(num_bins, dtype=np.float32)
  confidence_histogram = np.zeros(num_bins, dtype=np.float32)
  if example_weights is None:
    example_weights = np.ones_like(predictions, dtype=np.float32)
  bin_length = 1. / num_bins
  # The upper bound is half-open in np.arange, so we add a smaller quantity on
  # top to include 1 in the resulting bounds.
  bounds = np.arange(0, 1 + 1. / (num_bins + 1), bin_length)
  for prediction, label, confidence_score, weight in zip(
      predictions, labels, confidence_scores, example_weights):
    # np.searchsorted returns 0 for 0, but 1 for any epsilon > 0 that belongs to
    # the first bin, so we need to add a special case for explicit 0 confidence.
    bin_index = (np.searchsorted(bounds, confidence_score) - 1
                 if confidence_score > 0 else 0)
    bin_example_counts[bin_index] += weight
    confidence_histogram[bin_index] += weight * confidence_score
    if prediction == label:
      bin_accurate_predictions[bin_index] += weight
  accuracy_histogram = bin_accurate_predictions / bin_example_counts
  confidence_histogram /= bin_example_counts
  # Now replace the infinities where there was a division by zero with np.nan.
  empty_bins = np.where(bin_example_counts == 0)[0]
  accuracy_histogram[empty_bins] = np.nan
  confidence_histogram[empty_bins] = np.nan
  return accuracy_histogram, confidence_histogram, bin_example_counts


def risk_reliability_histogram(
    labels: ListOrArray,
    risk_scores: ListOrArray,
    num_bins=DEFAULT_NUM_BINS,
    example_weights: Optional[ListOrArray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes a histogram of risk vs positive rate for the specified # of bins.

  In the resulting histogram, the risk is the x axis (buckets) and the bin
  values correspond to the percentage of positives within each bucket. The
  reliability diagram/histogram is often used and has been discussed for
  instance in (DeGroot & Fienberg, 1983), (Niculescu-Mizil & Caruana, 2005),
  (Guo et al., 2017). One could define it alternatively via confidence-accuracy,
  but this is a more straightforward way of going from a risk to a calibration
  of risk

  Args:
    labels: The original data labels.
    risk_scores: The risk scores produced by the model.
    num_bins: The number of bins to use when computing the diagram.
    example_weights: If specified, the histogram will be computed from the
      weighted averages instead.

  Returns:
    (positive_rate_histogram, risk_histogram, counts) where:
      positive_rate_histogram is the positive rates corresponding to each bin
        amongst bins that reflect risk ranges from i/M to (i+1)/M.
      risk_histogram is the confidences corresponding to each bin amongst bins
        that reflect risk ranges from i/M to (i+1)/M.
      counts gives the number of examples per each bin.
  """
  bin_example_counts = np.zeros(num_bins, dtype=np.float32)
  bin_num_positives = np.zeros(num_bins, dtype=np.float32)
  risk_histogram = np.zeros(num_bins, dtype=np.float32)
  if example_weights is None:
    example_weights = np.ones_like(risk_scores, dtype=np.float32)
  bin_length = 1. / num_bins
  # The upper bound is half-open in np.arange, so we add a smaller quantity on
  # top to include 1 in the resulting bounds.
  bounds = np.arange(0, 1 + 1. / (num_bins + 1), bin_length)
  for label, risk_score, weight in zip(labels, risk_scores, example_weights):
    # np.searchsorted returns 0 for 0, but 1 for any epsilon > 0 that belongs to
    # the first bin, so we need to add a special case for explicit 0 risk.
    bin_index = (np.searchsorted(bounds, risk_score) - 1
                 if risk_score > 0 else 0)
    bin_example_counts[bin_index] += weight
    risk_histogram[bin_index] += weight * risk_score
    if label == 1:
      bin_num_positives[bin_index] += weight
  positive_rate_histogram = bin_num_positives / bin_example_counts
  risk_histogram /= bin_example_counts
  # Now replace the infinities where there was a division by zero with np.nan.
  empty_bins = np.where(bin_example_counts == 0)[0]
  positive_rate_histogram[empty_bins] = np.nan
  risk_histogram[empty_bins] = np.nan
  return positive_rate_histogram, risk_histogram, bin_example_counts


def expected_calibration_error(observed_score_histogram: ListOrArray,
                               predicted_score_histogram: ListOrArray,
                               bin_example_counts: ListOrArray) -> float:
  """Computes the expected calibration error for a model.

  Args:
    observed_score_histogram: A binned histogram of either accuracies or true
      positive rates (for the risk case) for equally-spaced confidence/risk
      bins.
    predicted_score_histogram: A binned histogram of confidence/risk scores for
      equally-spaced confidence/risk bins.
    bin_example_counts: The number of examples in each bin.

  Returns:
    The computed expected calibration error.
  """
  total_examples = np.sum(bin_example_counts)
  expected_error = np.nan
  for observed_value, predicted_value, bin_count in zip(
      observed_score_histogram, predicted_score_histogram,
      bin_example_counts):
    if not np.isnan(observed_value) and not np.isnan(predicted_value):
      # To avoid reporting a zero error for all-nan cases.
      if expected_error is np.nan:
        expected_error = 0
      expected_error += ((abs(observed_value - predicted_value) * bin_count)
                         / total_examples)
  return expected_error


def maximum_calibration_error(observed_score_histogram: np.ndarray,
                              predicted_score_histogram: np.ndarray) -> float:
  """Computes the maximum calibration error for a model.

  Args:
    observed_score_histogram: A binned histogram of either accuracies or true
      positive rates (for the risk case) for equally-spaced confidence/risk
      bins.
    predicted_score_histogram: A binned histogram of confidence/risk scores for
      equally-spaced confidence/risk bins.

  Returns:
    The computed maximum calibration error.
  """
  abs_diff = np.fabs(observed_score_histogram - predicted_score_histogram)
  return np.nanmax(abs_diff)


def reliability_diagram_from_posneg_lists(
    pos: ListOrArray,
    neg: ListOrArray,
    pos_weights: ListOrArray,
    neg_weights: ListOrArray,
    num_bins: int = DEFAULT_NUM_BINS,
    class_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes a histogram of confidence vs accuracy for the specified # of bins.

  In the resulting histogram, the confidence is the x axis (buckets) and the
  bin values correspond to the accuracy within each bucket. The reliability
  diagram/histogram is often used and has been discussed for instance in
  (DeGroot & Fienberg, 1983), (Niculescu-Mizil & Caruana, 2005),
  (Guo et al., 2017).

  Args:
    pos: List of positive example outputs.
    neg: List of negative example outputs.
    pos_weights: List of positive example weights.
    neg_weights: List of negative example weights.
    num_bins: The number of bins to use when computing the diagram.
    class_threshold: The values below the operating point are to be considered
        as negative predictions, the values above the threshold as the
        positive predictions.

  Returns:
    (accuracy_histogram, confidence_histogram, counts) where:
      accuracy_histogram is the accuracies corresponding to each bin amongst
        bins that reflect confidence ranges from i/M to (i+1)/M.
      confidence_histogram is the confidences corresponding to each bin amongst
        bins that reflect confidence ranges from i/M to (i+1)/M.
      counts gives the number of examples per each bin.
  """
  assert (class_threshold < 1) and (class_threshold > 0)
  labels = np.zeros(len(pos) + len(neg), dtype=np.int32)
  labels[:len(pos)] = 1
  soft_predictions = np.concatenate((pos, neg), axis=0)
  example_weights = np.concatenate((pos_weights, neg_weights),
                                   axis=0)
  predictions = (soft_predictions > class_threshold).astype(np.int32)
  confidences = []
  for x in soft_predictions:
    if x <= class_threshold:
      confidences.append(1. - (x * 0.5 / class_threshold))
    else:
      confidences.append(x * 0.5 / (1 - class_threshold) +
                         (0.5 - class_threshold) / (1 - class_threshold))
  confidences = np.array(confidences, dtype=np.float32)
  return reliability_histogram(
      predictions, labels, confidences, num_bins=num_bins,
      example_weights=example_weights)


def risk_reliability_diagram_from_posneg_lists(
    pos: ListOrArray,
    neg: ListOrArray,
    pos_weights: ListOrArray,
    neg_weights: ListOrArray,
    num_bins: int = DEFAULT_NUM_BINS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes a histogram of risk vs positive rate for the specified # of bins.

  In the resulting histogram, the risk is the x axis (buckets) and the bin
  values correspond to the percentage of positives within each bucket. The
  reliability diagram/histogram is often used and has been discussed for
  instance in (DeGroot & Fienberg, 1983), (Niculescu-Mizil & Caruana, 2005),
  (Guo et al., 2017). One could define it alternatively via confidence-accuracy,
  but this is a more straightforward way of going from a risk to a calibration
  of risk

  Args:
    pos: List of positive example outputs.
    neg: List of negative example outputs.
    pos_weights: List of positive example weights.
    neg_weights: List of negative example weights.
    num_bins: The number of bins to use when computing the diagram.

  Returns:
    (positive_rate_histogram, risk_histogram, counts) where:
      positive_rate_histogram is the positive rates corresponding to each bin
        amongst bins that reflect risk ranges from i/M to (i+1)/M.
      risk_histogram is the confidences corresponding to each bin amongst bins
        that reflect risk ranges from i/M to (i+1)/M.
      counts gives the number of examples per each bin.
  """
  labels = np.zeros(len(pos) + len(neg), dtype=np.int32)
  labels[:len(pos)] = 1
  risk_predictions = np.concatenate((pos, neg), axis=0)
  example_weights = np.concatenate((pos_weights, neg_weights),
                                   axis=0)
  return risk_reliability_histogram(
      labels, risk_predictions, num_bins=num_bins,
      example_weights=example_weights)


