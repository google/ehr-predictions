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
"""Loss function utility to be used in tasks."""

from typing import Optional

from ehr_prediction_modeling import types
import tensorflow.compat.v1 as tf


def loss_fn(y: tf.Tensor,
            targets: tf.Tensor,
            loss_mask: tf.Tensor,
            loss_type: str = types.TaskLossType.CE,
            scale_pos_weight: Optional[float] = None) -> tf.Tensor:
  """Return the batch loss vector masked by loss_mask.

  Args:
    y: Output of the model used to compare against targets, has shape
      [num_unroll, batch_size, channels, num_targets].
    targets: Ground-truth labels used to compare against y, has shape
      [num_unroll, batch_size, channels, num_targets].
    loss_mask: Mask to apply to loss, has shape [num_unroll, batch_size,
      channels, num_targets].
    loss_type: Type of loss function. Must be one of types.TaskLossType.
    scale_pos_weight: Weight to scale positive examples by.
  Returns:
    A loss graph variable.
  """
  # TODO(b/158257206): Add a test to verify the loss is computed appropriatly
  # without silent broadcasting.
  if loss_type == types.TaskLossType.CE:

    def _loss_function(x, y, z):
      return tf.losses.sigmoid_cross_entropy(
          x, y, weights=scale_pos_fn(x, z, scale_pos_weight))
  elif loss_type == types.TaskLossType.MULTI_CE:

    def _loss_function(x, y, z):
      return tf.losses.softmax_cross_entropy(
          x, y, weights=scale_pos_fn(x, z, scale_pos_weight))
  elif loss_type == types.TaskLossType.BRIER:

    def _loss_function(x, y, z):
      return tf.losses.mean_squared_error(
          x, tf.sigmoid(y), weights=scale_pos_fn(x, z, scale_pos_weight))
  elif loss_type == types.TaskLossType.L1:

    def _loss_function(x, y, z):
      x = tf.cast(x, tf.float32)
      y = tf.cast(y, tf.float32)
      z = tf.cast(z, tf.float32)
      return tf.losses.absolute_difference(x, y, weights=z)
  elif loss_type == types.TaskLossType.L2:

    def _loss_function(x, y, z):
      x = tf.cast(x, tf.float32)
      y = tf.cast(y, tf.float32)
      z = tf.cast(z, tf.float32)
      return tf.losses.mean_squared_error(x, y, weights=z)
  else:
    raise ValueError(f"Invalid loss_type: {loss_type}")

  targets.shape.assert_same_rank(y.shape)
  loss = _loss_function(targets, y, loss_mask)
  if loss.shape != tf.TensorShape(()):
    raise ValueError(f"Expected a scalar loss, found {loss}.")

  return loss


def scale_pos_fn(targets: tf.Tensor,
                 loss_mask: tf.Tensor,
                 scale_pos_weight: Optional[float] = None) -> tf.Tensor:
  """Re-weight positive examples by scale_pos_weight."""
  if scale_pos_weight is None:
    return loss_mask
  reweighted_targets = tf.cast(targets, tf.float32) * tf.constant(
      scale_pos_weight - 1, dtype=tf.float32) + 1.
  return tf.cast(loss_mask, dtype=tf.float32) * reweighted_targets


