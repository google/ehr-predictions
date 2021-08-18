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
"""Base class with variables and functionality common across deep encoders."""

import abc
import typing
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.models.nets import sparse_lookup
import sonnet as snt
import tensorflow.compat.v1 as tf

if typing.TYPE_CHECKING:
  from ehr_prediction_modeling import configdict

EncoderInput = Tuple[tf.SparseTensor, tf.SparseTensor]
EncoderContext = Optional[Dict[str, Union[tf.Tensor, bool, tf.SparseTensor]]]
DeepEncoderTypes = TypeVar("DeepEncoderTypes", bound="DeepEncoderBase")


class DeepEncoderBase(snt.AbstractModule):
  """Base encoder class for deep encoders with hidden layers.

  Specifically, used for Encoders that have SparseLookupEncoder initial layer.
  """

  def __init__(self,
               ndim_emb: int,
               ndim_input: int,
               emb_config: "configdict.ConfigDict",
               enc_config: "configdict.ConfigDict",
               n_encoder_hidden: Optional[List[int]] = None,
               n_act: Optional[int] = 0,
               act_fn: Optional[Callable[[tf.Tensor], tf.Tensor]] = tf.nn.tanh,
               name: Optional[str] = None,
               identity_lookup: Optional[bool] = False):

    if name is None:
      name = type(self).__name__

    super().__init__(name=name)

    self._ndim_emb = ndim_emb
    self._ndim_input = ndim_input
    self._emb_config = emb_config
    self._enc_config = enc_config
    self._n_act = n_act
    self._dropout_prob = self._enc_config.get("embedding_dropout_prob", 0.)
    self._dropout_is_training = types.ModelMode.is_train(
        self._enc_config.get("mode", types.ModelMode.TRAIN))
    self._use_batch_norm = self._emb_config.get("arch_args.use_batch_norm",
                                                False)
    self._act_fn = act_fn
    self._sparse_combine = self._enc_config.get("sparse_combine", "sum")
    self._n_encoder_hidden = n_encoder_hidden
    self._encoder_depth = len(self._n_encoder_hidden)
    self._activate_final = self._emb_config.get("arch_args.activate_final",
                                                False)
    self._use_highway_connection = self._emb_config.get(
        "arch_args.use_highway_connection", False)

    self.initial_embedding = sparse_lookup.SparseLookupEncoder(
        self.initial_embed_size,
        self._ndim_input,
        self._n_act,
        activate_final=False,
        act_fn=self._act_fn,
        sparse_lookup_dropout_prob=self._enc_config.get(
            "sparse_lookup_dropout_prob", 0.0),
        dropout_is_training=self._dropout_is_training,
        sparse_combine=self._sparse_combine,
        identity_lookup=identity_lookup)

  @property
  def embed_out(self) -> int:
    """Returns the output embedding size for pres and num features."""
    return self._ndim_emb

  @property
  def initial_embed_size(self) -> int:
    if self._encoder_depth > 0:
      initial_embed_size = self._n_encoder_hidden[0]
    else:
      initial_embed_size = self.embed_out
    return initial_embed_size

  @abc.abstractmethod
  def _build(self,
             input_data: EncoderInput,
             context: EncoderContext = None) -> tf.Tensor:
    """Builds the encoder.

    Args:
      input_data: A list of [indices, values] in this order, both in the
        SparseTensor representation.
      context : Optional context or data to be used for conditioning. This is
        not used by some networks.

    Returns:
      The embedding, a Tensor of size (batch x ndim_emb).
    """
