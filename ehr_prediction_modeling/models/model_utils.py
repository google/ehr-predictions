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
"""Utilities for model and encoders."""

import collections
import typing
from typing import Callable, Dict, List, Optional, Union

from absl import logging
from ehr_prediction_modeling import types
import tensorflow.compat.v1 as tf
import tf_slim as slim
import tree as nest

if typing.TYPE_CHECKING:
  from ehr_prediction_modeling import configdict


def get_regularizer(
    regularizer_type: str, l_reg_factor_weight: float
) -> Optional[Callable[[tf.Tensor], Optional[tf.Tensor]]]:
  """Gets a regularizer of a given type and scale.

  Args:
    regularizer_type: One of types.RegularizationType
    l_reg_factor_weight: Scale for regularization.

  Returns:
    A function with weights parameter that applies regularization.
  """
  if regularizer_type == types.RegularizationType.NONE:
    return None
  elif regularizer_type == types.RegularizationType.L1:
    return slim.l1_regularizer(scale=l_reg_factor_weight)
  elif regularizer_type == types.RegularizationType.L2:
    return slim.l2_regularizer(scale=l_reg_factor_weight)
  else:
    raise ValueError(f"Unknown regularization type {regularizer_type}")


def get_optimizer_from_config(
    optimizer_config: "configdict.ConfigDict") -> tf.train.Optimizer:
  """Instantiates the optimizer based on the optimizer config."""
  if optimizer_config.learning_rate_scheduling == (
      types.LearningRateScheduling.FIXED):
    learning_rate = optimizer_config.initial_learning_rate
  elif optimizer_config.learning_rate_scheduling == (
      types.LearningRateScheduling.EXPONENTIAL_DECAY):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        optimizer_config.initial_learning_rate,
        global_step,
        optimizer_config.lr_decay_steps,
        optimizer_config.lr_decay_base,
        staircase=True)
  else:
    raise ValueError("Unknown learning rate scheduling option "
                     f"{optimizer_config.learning_rate_scheduling}")
  return _get_optimizer(optimizer_config, learning_rate)


def _get_optimizer(
    optimizer_config: "configdict.ConfigDict",
    learning_rate: Union[float, tf.Tensor]) -> tf.train.Optimizer:
  """Returns the optimizer instance."""
  if optimizer_config.optim_type == types.Optimizers.SGD:
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_config.optim_type == types.Optimizers.ADAM:
    return tf.train.AdamOptimizer(learning_rate, optimizer_config.beta1,
                                  optimizer_config.beta2)
  elif optimizer_config.optim_type == types.Optimizers.RMSPROP:
    return tf.train.RMSPropOptimizer(learning_rate, optimizer_config.decay,
                                     optimizer_config.mom)
  else:
    raise ValueError(f"Unknown optimizer type {optimizer_config.optim_type}")


def optim_fn(optimizer: tf.train.Optimizer,
             loss: tf.Tensor,
             var_list: Optional[List[tf.Variable]] = None,
             norm_clip: bool = False,
             increment_step: bool = True) -> tf.Operation:
  """Returns the optimization step function and the global counter."""
  return multiple_loss_optim_fn(
      optimizer,
      loss_to_var_list={loss: var_list},
      norm_clip=norm_clip,
      increment_step=increment_step)


def multiple_loss_optim_fn(optimizer: tf.train.Optimizer,
                           loss_to_var_list: Dict[tf.Tensor,
                                                  Optional[List[tf.Variable]]],
                           norm_clip: bool = False,
                           increment_step: bool = True) -> tf.Operation:
  """Returns the optimization step function and the global counter.

  Computes gradients for a set of losses and their corresponding variables.

  Args:
    optimizer: Optimizer to be used for computing and applying gradients.
    loss_to_var_list: A dictionary from a loss value to a list of variables that
      are to be optimized wrt that loss.
    norm_clip: Whether to use clip_by_norm on the gradients.
    increment_step: Whether to increment global_step.

  Returns:
    The apply_gradients operations, including step counter increase.
  """
  logging.info("Using the following loss to variable list dictionary: %s",
               loss_to_var_list)
  step_cnt = tf.train.get_or_create_global_step()

  all_grads_and_vars = []
  for loss, var_list in loss_to_var_list.items():
    if not var_list:
      raise ValueError(
          "No var_list was found for one of the losses passed to the optimizer")
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    all_grads_and_vars.extend(grads_and_vars)

  grads_by_var = collections.defaultdict(list)
  for grad, var in all_grads_and_vars:
    grads_by_var[var].append(grad)
    if len(grads_by_var[var]) > 1:
      raise ValueError(
          f"Multiple gradients associated with the following variable: {var}")

  if norm_clip:
    for idx, (g, v) in enumerate(all_grads_and_vars):
      if "cell" in v.name.lower():
        all_grads_and_vars[idx] = (tf.clip_by_norm(
            g, tf.constant(norm_clip, tf.float32)), v)

  return optimizer.apply_gradients(
      all_grads_and_vars, global_step=(step_cnt if increment_step else None))


def all_elements_equal(arr: List[int]) -> bool:
  """Checks whether all elements in a list are equal."""
  return arr.count(arr[0]) == len(arr)


def rnn_step_with_state(
    model,
    initial_state: List[tf.Tensor],
    modify_state: bool,
    step_args=None,
    step_kwargs=None,
    state_scope_name: Optional[str] = None) -> types.ForwardFunctionReturns:
  """Step through RNN with persistent state."""
  step_args = step_args or {}
  step_kwargs = step_kwargs or {}

  def create_state(t):
    """Create and initialize a local variable with a value for persistence."""
    return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  # Creates a unique variable scope to ensure the variable name is unique.
  reuse = tf.AUTO_REUSE if state_scope_name is not None else None
  with tf.variable_scope(state_scope_name, default_name="state", reuse=reuse):
    prev_state_vars = nest.map_structure(create_state, initial_state)
  prev_state = nest.map_structure(lambda x: x.read_value(), prev_state_vars)
  outputs = model(prev_states=prev_state, *step_args, **step_kwargs)

  if modify_state:
    assign_states = nest.map_structure(lambda x, v: x.assign(v),
                                       prev_state_vars, outputs.hidden_states)
    with tf.control_dependencies(nest.flatten(assign_states)):
      new_outputs = nest.map_structure(
          lambda x: tf.identity(x) if isinstance(x, tf.Tensor) else x, outputs)
      # This is not ideal but necessary for preserving gradient flow between
      # logits and activations.
      new_outputs = new_outputs._replace(activations=outputs.activations)

    return new_outputs
  else:
    return outputs


class RNNModelWithPersistentState():
  """RNN with self contained persistent state, should only be called once."""

  def __init__(self,
               model,
               modify_state: bool = True,
               batch_size: Optional[int] = None):
    self._model = model
    self._modify_state = modify_state
    self._batch_size = batch_size or self._model.batch_size
    self._initial_state = self._model.initial_state(self._batch_size)
    self._called = False

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def get_model(self):
    return self._model

  def __call__(self, *args, **kwargs) -> types.ForwardFunctionReturns:
    if self._called:
      logging.warn("RNNModelWithPersistentState called more than once.")

    self._called = True
    return rnn_step_with_state(
        self._model,
        initial_state=self._initial_state,
        modify_state=self._modify_state,
        step_args=args,
        step_kwargs=kwargs)

  def __getattr__(self, name):
    return getattr(self._model, name)
