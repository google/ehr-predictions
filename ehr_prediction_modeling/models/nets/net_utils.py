# coding=utf-8
# Copyright 2020 Google Health Research.
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
"""Defines utilities used by modules in nets."""

from typing import Callable, Dict, Optional

import sonnet as snt
import tensorflow.compat.v1 as tf

_W_INITIALIZER = tf.random_normal_initializer(stddev=0.02)
_B_INITIALIZER = tf.zeros_initializer()
INITIALIZER_DICT = {"w": _W_INITIALIZER, "b": _B_INITIALIZER}


def res_block(x: tf.Tensor,
              nhidden: int,
              highway: bool,
              use_bias: bool = True) -> tf.Tensor:

  """Builds a residual layer to be chained."""
  h = snt.Linear(output_size=nhidden, use_bias=use_bias)(x)

  if highway:
    # If highway, then mix data and transform using convex combination
    w = snt.Linear(output_size=nhidden, initializers=INITIALIZER_DICT)(x)
    w = tf.sigmoid(w)
    out = tf.multiply(h, w) + tf.multiply(x, 1 - w)
  else:
    # Standard residual connection
    out = h + x


  return out


