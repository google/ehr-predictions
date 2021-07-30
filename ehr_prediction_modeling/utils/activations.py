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
"""Activations functions and factory."""
from typing import Callable, Optional
import tensorflow.compat.v1 as tf


def act_elish(x: tf.Tensor) -> tf.Tensor:
  """Exponential linear sigmoid squashing activation function.

  Initially proposed in: https://arxiv.org/abs/1808.00783

  Args:
    x: Input to the activation function.

  Returns:
    The computed activation.
  """
  positive = tf.greater_equal(x, 0)
  return (tf.to_float(positive) * act_swish(x)
          + (1 - tf.to_float(positive)) * (tf.exp(x) - 1) * tf.sigmoid(x))


def act_hard_elish(x: tf.Tensor) -> tf.Tensor:
  """Exponential linear sigmoid squashing activation function variant.

  Initially proposed in: https://arxiv.org/abs/1808.00783

  Args:
    x: Input to the activation function.

  Returns:
    The computed activation.
  """
  positive = tf.greater_equal(x, 0)
  return (tf.to_float(positive) * x * act_hard_sigmoid(x)
          + (1 - tf.to_float(positive)) * (tf.exp(x) - 1) * act_hard_sigmoid(x))


def act_hard_sigmoid(x: tf.Tensor) -> tf.Tensor:
  """The hard sigmoid activation function.

  Args:
    x: Input to the activation function.

  Returns:
    The computed activation.
  """
  return tf.maximum(0, tf.minimum(1, (x + 1) / 2))


def act_swish(x: tf.Tensor) -> tf.Tensor:
  """The Swish activation function, as per https://arxiv.org/abs/1710.05941.

  Args:
    x: Input to the activation function.

  Returns:
    The computed activation.
  """
  return x * tf.sigmoid(x)


def act_tanh_pen(x: tf.Tensor) -> tf.Tensor:
  """The penalized tanh activation function.

  Shown to be promising for NLP tasks in https://arxiv.org/abs/1901.02671

  Args:
    x: Input to the activation function.

  Returns:
    The computed activation.
  """
  positive = tf.greater_equal(x, 0)
  return (tf.to_float(positive) * tf.nn.tanh(x)
          + (1 - tf.to_float(positive)) * 0.25 * tf.nn.tanh(x))


def get_activation(
    act_fn: str,
    lrelu_coeff: Optional[float] = 0.2) -> Callable[[tf.Tensor], tf.Tensor]:
  """Check and supply activation function."""
  name_to_activation_fn = {
      "tanh": tf.nn.tanh,
      "tanh_pen": act_tanh_pen,
      "relu": tf.nn.relu,
      "lrelu": lambda x: tf.maximum(lrelu_coeff * x, x),
      "swish": act_swish,
      "elish": act_elish,
      "hard_elish": act_hard_elish,
      "elu": tf.nn.elu,
      "selu": tf.nn.selu,
      "sigmoid": tf.nn.sigmoid,
      "hard_sigmoid": act_hard_sigmoid,
  }

  if act_fn not in name_to_activation_fn:
    raise ValueError(f"Unknown activation function: {act_fn}")

  return name_to_activation_fn[act_fn]
