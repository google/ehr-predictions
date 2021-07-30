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
"""Lookup Embedding module."""

import typing
from typing import Dict

from ehr_prediction_modeling.embeddings import input_embedding_base
from ehr_prediction_modeling.models.nets import sparse_lookup
import tensorflow.compat.v1 as tf


if typing.TYPE_CHECKING:
  from ehr_prediction_modeling import configdict


class BasicEmbeddingLookup(input_embedding_base.InputEmbeddingBase):
  """The class that uses the base embedding lookup."""

  def __init__(self, encoder_config: "configdict.ConfigDict",
               emb_dim_dict: Dict[str, int]):
    """Initialize the embedding base object."""
    super().__init__(encoder_config, emb_dim_dict)

    with tf.variable_scope(None, default_name=self._name) as variable_scope:
      self.variable_scope = variable_scope
      # Initialize all the weights.
      self._initialize_weights()

  def _initialize_weights(self):
    """Initialize the weights for embedding computations."""
    for feat_type in self._all_features:
      self._encoders[feat_type] = sparse_lookup.SparseLookupEncoder(
          ndim_emb=self._embed_dim_dict[feat_type],
          ndim_input=self._config.ndim_dict[feat_type],
          n_act=self._config.nact_dict[feat_type],
          sparse_lookup_dropout_prob=self._config.get(
              "sparse_lookup_dropout_prob", 0.0),
          dropout_is_training=self._is_training,
          sparse_combine=self._config.sparse_combine,
          name="lookup_embedding_" + feat_type,
          identity_lookup=feat_type in self._config.identity_lookup_features)
