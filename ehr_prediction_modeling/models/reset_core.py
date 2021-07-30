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
"""Wrapper to reset core states on specific steps within batched unrolls."""

from typing import List, Tuple, TypeVar
import sonnet as snt
import tensorflow.compat.v1 as tf
import tree as nest

RNNCore = TypeVar("RNNCore", bound=tf.nn.rnn_cell.RNNCell)


class ResetCore(snt.RNNCore):
  """A wrapper for managing state resets during unrolls.

  When unrolling an `RNNCore` on a batch of inputs sequences it may be necessary
  to reset the core's state at different timesteps for different elements of the
  batch. The `ResetCore` class enables this by taking a batch of `should_reset`
  booleans in addition to the batch of inputs, and conditionally resetting the
  core's state for individual elements of the batch.
  """

  def __init__(self, core: RNNCore):
    """Constructor of the `ResetCore` wrapper.

    Args:
      core: `snt.RNNCore`-like core to be wrapped.
    """
    super().__init__(name="reset_core")
    self._core = core

  def _build(self, input_should_reset: Tuple[tf.Tensor, tf.Tensor],
             state: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
    """Implements the conditional resetting logic in-graph.

    For each element of the batch, `core` is applied using the previous core
    state by default, but resetting the state when `should_reset` signals
    that the specific element of the batch should have the state reset.

    Args:
      input_should_reset: A pair of tensors containing respectively a
        batch of inputs to the `core`, and a batch of reset signals (a boolean
        tensor of shape [batch_size, 1]).
      state: A batch of previous states of the `core`.
      **kwargs: additional kwargs to pass to the `_build` method of the `core`.

    Returns:
      the core output and new state.
    """
    (input_, should_reset) = input_should_reset

    dtype = nest.flatten(input_)[0].dtype
    if should_reset.shape.is_fully_defined():
      # Prevent recomputing the initial state when unnecessary.
      with tf.control_dependencies(None):
        initial_state = self.initial_state(should_reset.shape[0], dtype=dtype)
    else:
      initial_state = self.initial_state(tf.shape(should_reset)[0], dtype=dtype)

    # Use a reset state for the selected elements in the batch.
    should_reset = tf.squeeze(should_reset, -1)
    state = nest.map_structure(
        lambda i, s: tf.where(should_reset, i, s), initial_state, state)
    return self._core(input_, state, **kwargs)

  @property
  def wrapped_core(self) -> RNNCore:
    """The wrapped core."""
    return self._core

  @property
  def state_size(self) -> tf.TensorShape:
    """Forward size(s) of state(s) used by the wrapped core."""
    return self._core.state_size

  @property
  def output_size(self) -> tf.TensorShape:
    """Forward size of outputs produced by the wrapped core."""
    return self._core.output_size

  @property
  def trainable_weights(self) -> List[tf.Tensor]:
    """Trainable weights of the wrapped core."""
    return self._core.trainable_weights

  def initial_state(self, *args, **kwargs) -> List[tf.Tensor]:
    """Creates the core initial state."""
    return self._core.zero_state(*args, **kwargs)
