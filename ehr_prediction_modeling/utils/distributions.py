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

"""Wrappers for probability distributions needed for EHR modelling."""

from typing import List, Optional

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


class HardConcrete(tfp.distributions.Distribution):
  """Hard Concrete distribution. https://arxiv.org/abs/1712.01312."""

  def __init__(
      self,
      logits: tf.Tensor,
      temperature: tf.Tensor,
      lower: tf.Tensor,
      higher: tf.Tensor,
      validate_args: bool = False,
      allow_nan_stats: bool = True,
      name: str = "HardConcrete",
  ):
    """Create a HardConcrete distribution.

    Parameter validation are performed by self._parameter_control_dependencies.

    Args:
      logits: An N-D `Tensor` representing the log-odds of a positive event. In
        the paper, this corresponds to the parameter `log(alpha)`.
      temperature: The temperature value. Must have `temperature > 0.0`. Beta in
        the paper.
      lower: The lower boundary of the support interval. Must have `lower <
        higher` and `lowers < 0.0`. Gamma in the paper.
      higher: The upper boundary of the support interval. Must have `lower <
        higher` and `higher > 1.0`. Zeta in the paper
      validate_args: When `True` distribution parameters are checked for
        validity despite possibly degrading runtime performance.
      allow_nan_stats: When `True`, statistics (e.g., mean, mode, variance) use
        the value "`NaN`" to indicate the result is undefined. When `False`, an
        exception is raised if one or more of the statistic's batch members are
        undefined.
      name: Name prefixed to Ops created by this class.

    Raises:
      ValueError if temperature, lower or higher have rank > 0.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._logits = tf.identity(logits, name="logits")
      self._temperature = tf.cast(
          temperature, dtype=logits.dtype, name="temperature")
      self._lower = tf.cast(lower, dtype=logits.dtype, name="lower")
      self._higher = tf.cast(higher, dtype=logits.dtype, name="higher")

      if self._temperature.shape.rank > 0:
        raise ValueError(
            f"temperature must be a scalar, found {self._temperature}")
      if self._lower.shape.rank > 0:
        raise ValueError(f"lower must be a scalar, found {self._lower}")
      if self._higher.shape.rank > 0:
        raise ValueError(f"higher must be a scalar, found {self._higher}")

      super().__init__(
          dtype=logits.dtype,
          reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def l0_norm(self) -> tf.Tensor:
    """Returns the L0 norm."""
    offsets = self._temperature * tf.log(-self._lower / self._higher)
    return tf.reduce_sum(self._logits - offsets)

  def hard_sigmoid(self) -> tf.Tensor:
    """Computes the hard Sigmoid gate at test time."""
    # As described in the paper. Why is temperature not accounted for?
    return self._hard_sigmoid(self._logits)

  def _hard_sigmoid(self, logits: tf.Tensor) -> tf.Tensor:
    """Stretches and clips sigmoid samples between 0 and 1."""
    samples = tf.nn.sigmoid(logits) * (self._higher - self._lower) + self._lower
    return tf.clip_by_value(samples, 0., 1.)

  def _sample_n(
      self,
      num_samples: tf.Tensor,
      seed: Optional[int] = None,
  ) -> tf.Tensor:
    """Samples from the HardConcrete distribution."""
    shape = tf.concat([[num_samples], tf.shape(self._logits)], axis=0)
    noises = self._sample_logistic(shape, seed=seed)
    samples = (noises + self._logits) / self._temperature
    return self._hard_sigmoid(samples)

  def _sample_logistic(
      self,
      sample_shape: tf.Tensor,
      seed: Optional[int] = None,
  ) -> tf.Tensor:
    """Returns log(u) - log(1 - u) where u is from a Uniform (0, 1)."""
    dist = tfp.distributions.Logistic(loc=0., scale=1.)
    return dist.sample(sample_shape, seed=seed)

  def _parameter_control_dependencies(
      self,
      is_init: bool,
  ) -> List[tf.Operation]:
    if not self.validate_args:
      return []

    assertions = []
    assertions.append(
        tf.debugging.assert_less(
            0.0,
            self._temperature,
            message=("support not defined when `temperature` <= `0.0`: "
                     f"{self._temperature} <= 0.0.")))
    assertions.append(
        tf.debugging.assert_less(
            self._lower,
            self._higher,
            message=("support not defined when `lower` >= `higher`: "
                     f"{self._lower} >= {self._higher}.")))
    assertions.append(
        tf.debugging.assert_less(
            self._lower,
            0.0,
            message=("support not defined when `lower` >= `0.0`: "
                     f"{self._lower} >= 0.0.")))
    # Support is actually defined, but added for symmetry with lower.
    assertions.append(
        tf.debugging.assert_less(
            1.0,
            self._higher,
            message=("support not defined when `higher` <= `1.0`: "
                     f"{self._higher} >= 1.0.")))

    return assertions
