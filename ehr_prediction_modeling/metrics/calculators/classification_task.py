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
"""Metric calculator for classification tasks."""
import math
from typing import List, Mapping, Optional, Union
from ehr_prediction_modeling.metrics import calibration
from ehr_prediction_modeling.metrics import pr_roc
from ehr_prediction_modeling.metrics.calculators import base
import numpy as np

CONFIDENCE_CALIBRATION_HISTOGRAM_KEY = "full_confidence_calibration_histogram"
RISK_CALIBRATION_HISTOGRAM_KEY = "full_risk_calibration_histogram"

METRICS_EPSILON = 1e-5
ListOrArray = Union[List[Union[float, int]], np.ndarray]


class ClassificationTaskMetricCalculator(base.MetricCalculatorBase):
  """Calculator class for evaluation metrics for classification tasks."""

  ALL_METRICS = [
      "true_positives", "false_positives", "true_negatives", "false_negatives",
      "rocauc", "prauc", "normalised_prauc", "nb_ex", "nb_ex_pos", "nb_ex_neg",
      "tp", "tn", "ppv", "npv", "acc", "f1", "f2", "mcc",
      "expected_confidence_calibration_error",
      "maximum_confidence_calibration_error",
      CONFIDENCE_CALIBRATION_HISTOGRAM_KEY, "expected_risk_calibration_error",
      "maximum_risk_calibration_error", RISK_CALIBRATION_HISTOGRAM_KEY
  ]

  def __init__(self,
               pos: ListOrArray,
               neg: ListOrArray,
               pos_weights: ListOrArray,
               neg_weights: ListOrArray,
               num_calibration_bins: int = calibration.DEFAULT_NUM_BINS,
               class_threshold: float = 0.5) -> None:
    """Initialize the metrics calculator.

    Args:
      pos: List of positive example outputs.
      neg: List of negative example outputs.
      pos_weights: List of positive example weights.
      neg_weights: List of negative example weights.
      num_calibration_bins: The number of bins to use for the reliability
        diagram and the computation of the expected/max miscalibration.
      class_threshold: The point that splits the soft predictions into positives
        and negatives.
    """
    self._pos = pos
    self._neg = neg
    self._pos_weights = pos_weights
    self._neg_weights = neg_weights
    self._num_calibration_bins = num_calibration_bins
    self._class_threshold = class_threshold
    self._precompute_confusion_matrix()
    self._precompute_reliability_diagrams()

  def _precompute_reliability_diagrams(self) -> None:
    """Computes a set of histograms.

    These are for accuracy vs confidence and risk vs positive rate.
    """
    (self._accuracy_histogram, self._confidence_histogram,
     self._confidence_bin_counts) = (
         calibration.reliability_diagram_from_posneg_lists(
             self._pos,
             self._neg,
             self._pos_weights,
             self._neg_weights,
             num_bins=self._num_calibration_bins,
             class_threshold=self._class_threshold))
    (self._positive_rate_histogram, self._risk_histogram,
     self._risk_bin_counts) = (
         calibration.risk_reliability_diagram_from_posneg_lists(
             self._pos,
             self._neg,
             self._pos_weights,
             self._neg_weights,
             num_bins=self._num_calibration_bins))

  def _precompute_confusion_matrix(self) -> None:
    """Compute and store the true/false pos/neg rates."""
    # Since pos/neg are lists of positive and negative examples, this is how the
    # TP/TN/FP/FN map onto the (weighted) variables below:
    # TP: w_pos_true
    # TN: w_neg_true
    # FN: w_pos_wrong (positive examples, wrong prediction)
    # FP: w_neg_wrong (negative examples, wrong_prediction)
    self._w_pos_true = np.sum(
        np.multiply((np.asarray(self._pos) > self._class_threshold),
                    np.asarray(self._pos_weights)))
    self._w_neg_true = np.sum(
        np.multiply((np.asarray(self._neg) <= self._class_threshold),
                    np.asarray(self._neg_weights)))
    self._w_pos_wrong = np.sum(
        np.multiply((np.asarray(self._pos) <= self._class_threshold),
                    np.asarray(self._pos_weights)))
    self._w_neg_wrong = np.sum(
        np.multiply((np.asarray(self._neg) > self._class_threshold),
                    np.asarray(self._neg_weights)))

  def true_positives(self) -> int:
    """Compute the number of true positives."""
    return self._w_pos_true

  def false_positives(self) -> int:
    """Compute the number of false positives."""
    return self._w_neg_wrong

  def true_negatives(self) -> int:
    """Compute the number of true negatives."""
    return self._w_neg_true

  def false_negatives(self) -> int:
    """Compute the number of false negatives."""
    return self._w_pos_wrong

  def full_confidence_calibration_histogram(self) -> Mapping[str, np.ndarray]:
    """Gives the full confidence vs accuracy calibration histogram."""
    return {
        "accuracy_histogram": self._accuracy_histogram,
        "confidence_histogram": self._confidence_histogram,
        "confidence_bin_counts": self._confidence_bin_counts
    }

  def full_risk_calibration_histogram(self) -> Mapping[str, np.ndarray]:
    """Gives the full risk vs positive rate calibration histogram."""
    return {
        "positive_rate_histogram": self._positive_rate_histogram,
        "risk_histogram": self._risk_histogram,
        "risk_bin_counts": self._risk_bin_counts
    }

  def expected_confidence_calibration_error(self) -> float:
    """Compute the expected confidence calibration error."""
    return calibration.expected_calibration_error(self._accuracy_histogram,
                                                  self._confidence_histogram,
                                                  self._confidence_bin_counts)

  def maximum_confidence_calibration_error(self) -> float:
    """Compute the maximum confidence calibration error."""
    return calibration.maximum_calibration_error(self._accuracy_histogram,
                                                 self._confidence_histogram)

  def expected_risk_calibration_error(self) -> float:
    """Compute the expected risk calibration error."""
    return calibration.expected_calibration_error(self._positive_rate_histogram,
                                                  self._risk_histogram,
                                                  self._risk_bin_counts)

  def maximum_risk_calibration_error(self) -> float:
    """Compute the maximum risk calibration error."""
    return calibration.maximum_calibration_error(self._positive_rate_histogram,
                                                 self._risk_histogram)

  def rocauc(self) -> float:
    """Get the AUC from the given predictions."""
    return pr_roc.weighted_area_under_curve_from_raw_scores(
        self._pos, self._neg, self._pos_weights, self._neg_weights)

  def prauc(self) -> float:
    """Get the PRAUC from the given predictions."""
    return pr_roc.weighted_pr_area_under_curve_from_raw_scores(
        self._pos, self._neg, self._pos_weights, self._neg_weights)

  def normalised_prauc(self) -> Optional[float]:
    """Gets the normalised PRAUC (https://arxiv.org/pdf/1206.4667.pdf).

    Returns:
      None if examples are weighted; the normalised PRAUC does not extend
      to the weighted case.
    """
    all_weights = list(self._pos_weights) + list(self._neg_weights)
    # If weights are differing, return None
    if len(set(all_weights)) != 1:
      return None
    prauc = self.prauc()
    skew = self.nb_ex_pos() / float(self.nb_ex())
    # Handle degenerate cases as per the technical report to the cited paper.
    if self.nb_ex_pos() == 0:
      prauc_min = 0
    elif self.nb_ex_pos() == self.nb_ex():
      prauc_min = 1
    else:
      prauc_min = 1 + ((1 - skew) * math.log(1 - skew)) / skew
    return (prauc - prauc_min) / (1 - prauc_min)

  def nb_ex(self) -> int:
    """Get the total number of examples over which to compute metrics."""
    return np.sum(self._pos_weights) + np.sum(self._neg_weights)

  def nb_ex_pos(self) -> int:
    """Get the total number of positive examples in the computations."""
    return np.sum(self._pos_weights)

  def nb_ex_neg(self) -> int:
    """Get the total number of negative examples in the computations."""
    return np.sum(self._neg_weights)

  def tp(self) -> int:
    """Compute the sensitivity of the predictions."""
    return self._w_pos_true / np.sum(self._pos_weights)

  def tn(self) -> int:
    """Compute the specificity of the predictions."""
    return self._w_neg_true / np.sum(self._neg_weights)

  def ppv(self) -> float:
    """Compute the positive predictive value of the predictions."""
    return self._w_pos_true / (
        self._w_pos_true + self._w_neg_wrong + METRICS_EPSILON)

  def npv(self) -> float:
    """Compute the negative predictive value of the predictions."""
    return self._w_neg_true / (
        self._w_neg_true + self._w_pos_wrong + METRICS_EPSILON)

  def acc(self) -> float:
    """Compute the accuracy of the predictions."""
    return (self._w_pos_true + self._w_neg_true) / float(self.nb_ex())

  def f1(self) -> float:
    """Compute the F1 score of the predictions."""
    # F1 = 2 * pr * re / (pr + re). The formula below expresses it directly in
    # terms of TP/FP/TN/FN (https://en.wikipedia.org/wiki/F1_score).
    return (2. * self._w_pos_true /
            (2. * self._w_pos_true + self._w_pos_wrong + self._w_neg_wrong))

  def f2(self) -> float:
    """Compute the F2 score of the predictions."""
    # F2 might also be interesting as it increases the cost of false negatives,
    # which is useful for the task of deterioration prediction (AKI) as the cost
    # is indeed asymmetrical.
    return (
        5. * self._w_pos_true /
        (5. * self._w_pos_true + 4. * self._w_pos_wrong + self._w_neg_wrong))

  def mcc(self) -> float:
    """Compute the Matthews Correlation Coefficient for the predicitons.

    MCC is a useful metric for imbalanced datasets. It is a correlation
    coefficient between the observed and predicted (binary) classifications.

    NOTE: If we move to the multiclass case, this is also possible to generalise
    as an Rk statistic.

    Returns:
      The value of the MCC for the provided predictions.
    """
    mcc_denominator = math.sqrt((self._w_pos_true + self._w_neg_wrong) *
                                (self._w_pos_true + self._w_pos_wrong) *
                                (self._w_neg_true + self._w_neg_wrong) *
                                (self._w_neg_true + self._w_pos_wrong))
    if not mcc_denominator:
      return 0.
    mcc_numerator = (
        self._w_pos_true * self._w_neg_true -
        self._w_neg_wrong * self._w_pos_wrong)
    return mcc_numerator / mcc_denominator
