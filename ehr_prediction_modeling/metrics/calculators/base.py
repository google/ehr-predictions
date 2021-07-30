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
"""Base calculator class for computing metrics."""

from typing import Any, Mapping


class Error(Exception):
  """Base error class for ths file."""


class InvalidClassThresholdError(Error):
  """Indicates that a class threshold is not in the range (0,1)."""


class MetricCalculatorBase():

  ALL_METRICS = []

  def get_metrics_dict(self) -> Mapping[str, Any]:
    """Compute and return all metrics as a serializable dict."""
    return {metric: getattr(self, metric)() for metric in self.ALL_METRICS}
