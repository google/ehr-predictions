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
"""Definitions for namedtuples and enums used throughout experiments."""
import collections


class ForwardFunctionReturns(
    collections.namedtuple(
        "ForwardFunctionReturns",
        ["model_output", "hidden_states", "inputs", "activations"],
    )):
  """The variables returned by the forward function."""


class EmbeddingType(object):
  """Available embedding types."""
  LOOKUP = "Lookup"
  DEEP_EMBEDDING = "DeepEmbedding"


class EmbeddingEncoderType(object):
  """Available encoders for DeepEmbedding."""
  RESIDUAL = "residual"
  FC = "fc"
  SNR = "snr"


class ActivationFunction(object):
  """Available activation functions for cells in lstm_model."""
  TANH = "tanh"
  RELU = "relu"
  LRELU = "lrelu"
  SWISH = "swish"
  ELU = "elu"
  SLU = "selu"
  ELISH = "elish"
  HARD_ELISH = "hard_elish"
  SIGMOID = "sigmoid"
  HARD_SIGMOID = "hard_sigmoid"
  TANH_PEN = "tanh_pen"


class RegularizationType(object):
  """Available types of regularization for model and embedding weights."""
  NONE = "None"
  L1 = "L1"
  L2 = "L2"


class TaskType(object):
  """Available task types."""
  BINARY_CLASSIFICATION = "BinaryClassification"
  REGRESSION = "Regression"


class BestModelMetric(object):
  PRAUC = "prauc"
  ROCAUC = "rocauc"
  L1 = "l1"


class TaskNames(object):
  """Names of available tasks."""
  ADVERSE_OUTCOME_RISK = "AdverseOutcomeRisk"
  LAB_REGRESSION = "Labs"
  BINARIZED_LOS = "BinarizedLengthOfStay"
  REGRESSION_LOS = "RegressionLengthOfStay"
  MORTALITY = "MortalityRisk"
  READMISSION = "ReadmissionRisk"


class TaskTypes(object):
  """Type of tasks available for modeling."""
  ADVERSE_OUTCOME_RISK = "ao_risk"
  LAB_REGRESSION = "labs_regression"
  LOS = "length_of_stay"
  MORTALITY = "mortality"
  READMISSION = "readmission"


class TaskLossType(object):
  CE = "CE"
  MULTI_CE = "MULTI_CE"
  BRIER = "Brier"
  L1 = "L1"
  L2 = "L2"


class RNNCellType(object):
  """Available cell types for rnn_model."""
  SRU = "SRU"
  LSTM = "LSTM"
  UGRNN = "UGRNN"


class FeatureTypes(object):
  """Types of features available in the data representation."""
  PRESENCE_SEQ = "pres_s"  # Sequential presence features
  NUMERIC_SEQ = "num_s"  # Sequential numerical features
  CATEGORY_COUNTS_SEQ = "count_s"  # Sequential category counts features


# Mappings from feature names in the model to names in tf.SequenceExample.
FEAT_TO_NAME = {
    "pres_s": "presence",
    "num_s": "numeric",
    "count_s": "category_counts",
}


class Optimizers(object):
  SGD = "SGD"
  ADAM = "Adam"
  RMSPROP = "RMSprop"


class LearningRateScheduling(object):
  FIXED = "fixed"
  EXPONENTIAL_DECAY = "exponential_decay"


class EmbeddingCombinationMethod(object):
  """Available embedding combination methods."""
  CONCATENATE = "concatenate"
  SUM_ALL = "sum_all"
  SUM_BY_SUFFIX = "sum_by_suffix"
  COMBINE_SNR_OUT = "combine_snr_out"


class SNRBlockConnType(object):
  """Available unit types for SNREncoder."""
  FC = "fc"
  HIGHWAY = "highway"
  RESIDUAL = "residual"
  NONE = "None"


class SubNettoSubNetConnType(object):
  """Available connection types between subnetworks."""
  BOOL = "bool"
  SCALAR_WEIGHT = "scalar_weight"


class SNRInputCombinationType(object):
  """Available combination methods for subnetwork input."""
  CONCATENATE = "concatenate"
  SUM_ALL = "sum"


class LossCombinationType(object):
  SUM_ALL = "SUM_ALL"
  UNCERTAINTY_WEIGHTED = "UNCERTAINTY_WEIGHTED"


class ModelTypes(object):
  RNN = "RNN"
  SNRNN = "SNRNN"


class TaskLayerTypes(object):
  NONE = "none"
  MLP = "MLP"
  SNRMLP = "SNRMLP"


class ModelMode(object):
  """Available modes for the model."""
  TRAIN = "train"
  EVAL = "eval"
  PREDICT = "predict"

  @classmethod
  def is_train(cls, mode: str) -> bool:
    return mode == ModelMode.TRAIN

