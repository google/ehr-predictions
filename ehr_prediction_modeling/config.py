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
"""Default configuration."""

from ehr_prediction_modeling import types
from ehr_prediction_modeling.tasks import adverse_outcome_task
from ehr_prediction_modeling.tasks import labs_task
from ehr_prediction_modeling.tasks import los_task
from ehr_prediction_modeling.tasks import mortality_task
from ehr_prediction_modeling.tasks import readmission_task
from ehr_prediction_modeling import configdict

# Mapping of task type to Class.
TASK_MAPPING = {
    adverse_outcome_task.AdverseOutcomeRisk.task_type:
        adverse_outcome_task.AdverseOutcomeRisk,
    labs_task.LabsRegression.task_type:
        labs_task.LabsRegression,
    los_task.LengthOfStay.task_type:
        los_task.LengthOfStay,
    mortality_task.MortalityRisk.task_type:
        mortality_task.MortalityRisk,
    readmission_task.ReadmissionRisk.task_type:
        readmission_task.ReadmissionRisk
}


def get_config():
  """Gets configuration parameters.

  Returns:
    ConfigDict of default hyperparameters.
  """
  config = configdict.ConfigDict()

  shared = shared_config()
  config.shared = shared

  # Data splits over which to run evaluation.
  config.splits_to_evaluate = "valid"

  # Config for dataset reading and parsing.
  config.data = get_data_config(config.shared)

  # Config for saving checkpoints.
  config.checkpoint = get_checkpoint_config()

  # Setting the seed for the random initializations.
  config.seed = None

  config.tasks = shared.tasks


  task_configs = {}
  for task_factory in TASK_MAPPING.values():
    for task_config in task_factory.default_configs():
      task_configs[task_config.name] = task_config
  # Dict of prediction tasks configurations.
  config.task_configs = task_configs

  # Config for model parameters.
  config.model = get_model_config(shared)

  # Config for the encoder module to embed the data.
  config.encoder = get_encoder_config(shared)

  # Config for the optimizer.
  config.optimizer = get_optimizer_config()

  return config


def get_data_config(shared):
  """Gets config for data loading that is common to all experiments."""

  data_config = configdict.ConfigDict()

  # Add path to directory containing tfrecords.
  data_config.records_dirpath = (
    "ehr_prediction_modeling/fake_data/standardize")

  data_config.train_filename = "train.tfrecords"

  data_config.valid_filename = "valid.tfrecords"

  data_config.test_filename = "test.tfrecords"

  data_config.calib_filename = "calib.tfrecords"

  # Whether to shuffle the dataset.
  data_config.shuffle = True

  # The size of the shuffle buffer.
  data_config.shuffle_buffer_size = 64

  # Parallelism settings for the padded data reading implementation
  data_config.padded_settings = configdict.ConfigDict()

  # Number of parallel calls to parsing.
  data_config.padded_settings.parse_cycle_length = 4

  # Number of parallel recordio reads.
  data_config.padded_settings.recordio_cycle_length = 16

  # Number of (batched) sequences to prefetch.
  data_config.padded_settings.num_prefetch = 16

  # Extend config for dataset reading and parsing.
  data_config.context_features = shared.context_features
  data_config.sequential_features = shared.sequential_features
  data_config.batch_size = shared.batch_size
  data_config.num_unroll = shared.num_unroll
  data_config.segment_length = shared.segment_length

  return data_config


def get_model_config(shared):
  """Gets model configuration."""

  model_config = configdict.ConfigDict()

  # One of types.ModelModes
  model_config.mode = shared["mode"]

  # Number of training steps.
  model_config.num_steps = 400000

  # List of number of dimensions of the hidden layers of the rnn model.
  # If using types.ModelTypes.SNRNN the input expected is a list of lists where
  # each inner list would have the state dimensions for each cell in the layer.
  model_config.ndim_lstm = [200, 200, 200]

  # RNN activation function: one of types.ActivationFunctions
  model_config.act_fn = types.ActivationFunction.TANH

  # Name of the TensorFlow variable scope for model variables.
  model_config.scope = "model"

  # Whether or not to use the highway connections in the RNN.
  model_config.use_highway_connections = True

  # The choice of None / L1-regularization / L2-regularization for LSTM weights.
  # One of types.RegularizationType.
  model_config.l_regularization = types.RegularizationType.NONE

  # The choice of None / L1-regularization / L2-regularization for logistic
  # weights. The weight applied to these is task specific. One of
  # types.RegularizationType.
  model_config.logistic_l_regularization = types.RegularizationType.NONE

  # The weight used in L1/L2 regularization for LSTM weights.
  model_config.l_reg_factor_weight = 0.

  # Coefficient for leaky relu activation functions.
  model_config.leaky_relu_coeff = 0.2

  # Cell type for the model: one of types.RNNCellType
  model_config.cell_type = types.RNNCellType.SRU

  # Number of steps for which event sequence will be unrolled.
  model_config.num_unroll = shared.num_unroll

  # Batch size
  model_config.batch_size = shared.batch_size

  # Number of parallel iterations for dynamic RNN.
  model_config.parallel_iterations = 1

  model_config.cell_config = get_cell_config()

  model_config.snr = get_model_snr_config()

  return model_config


def get_optimizer_config():
  """Gets configuration for optimizer."""
  optimizer_config = configdict.ConfigDict()
  # Learning rate scheduling. One of: ["fixed", "exponential_decay"]
  optimizer_config.learning_rate_scheduling = "exponential_decay"

  # Optimization algorithm. One of: ["SGD", "Adam", "RMSprop"].
  optimizer_config.optim_type = "Adam"

  # Adam beta1.
  optimizer_config.beta1 = 0.9

  # Adam beta2.
  optimizer_config.beta2 = 0.999

  # Norm clipping threshold applied for rnn cells (no clip if 0).
  optimizer_config.norm_clip = 0.0

  # Learning rate.
  optimizer_config.initial_learning_rate = 0.001

  # The learning rate decay 'epoch' length.
  optimizer_config.lr_decay_steps = 12000

  # The learning rate decay base, applied per epoch.
  optimizer_config.lr_decay_base = 0.85

  # RMSprop decay.
  optimizer_config.decay = 0.9

  # RMSprop moment.
  optimizer_config.mom = 0.0

  return optimizer_config


def get_encoder_config(shared):
  """Gets config for encoder module that embeds data."""
  encoder_config = configdict.ConfigDict()
  encoder_config.sequential_features = shared.sequential_features
  encoder_config.context_features = shared.context_features
  encoder_config.identity_lookup_features = shared.identity_lookup_features

  # Name of the TensorFlow variable scope for encoder variables.
  encoder_config.scope = "encoder"

  # Number of dimensions of the embedding layer.
  encoder_config.ndim_emb = 400

  # Dict of median number of active features for each feature type. Updated
  # at runtime.
  encoder_config.nact_dict = {
      types.FeatureTypes.PRESENCE_SEQ: 10,
      types.FeatureTypes.NUMERIC_SEQ: 3,
      types.FeatureTypes.CATEGORY_COUNTS_SEQ: 3,
  }

  encoder_config.ndim_dict = {
      types.FeatureTypes.PRESENCE_SEQ: 142,
      types.FeatureTypes.NUMERIC_SEQ: 11,
      types.FeatureTypes.CATEGORY_COUNTS_SEQ: 10,
  }

  # How to combine the initial sparse multiplication. Valid options are
  # ["sum" (default), "sqrtn", "mean"]
  encoder_config.sparse_combine = "sum"

  # Number of steps for which event sequence will be unrolled.
  encoder_config.num_unroll = shared.num_unroll

  # The batch size.
  encoder_config.batch_size = shared.batch_size

  # How to combine embeddings. One of types.EmbeddingCombinationMethod
  encoder_config.embedding_combination_method = (
      types.EmbeddingCombinationMethod.CONCATENATE)

  # Embedding type enum as per types.EmbeddingType.
  encoder_config.embedding_type = types.EmbeddingType.DEEP_EMBEDDING

  # Probability of performing embedding dropout. Will not be applied to sparse
  # lookup layers. If types.EmbeddingType.LOOKUP is used, this value will be
  # ignored.
  encoder_config.embedding_dropout_prob = 0.0

  # Probability of performing embedding dropout on the sparse lookup layer.
  # If types.EmbeddingType.LOOKUP is used, this is the only dropout_prob that
  # will be used, embedding_dropout_prob will be ignored.
  encoder_config.sparse_lookup_dropout_prob = 0.0

  # Coefficient for leaky relu activation functions.
  encoder_config.leaky_relu_coeff = 0.2

  # The choice of None / L1-regularization / L2-regularization for the sparse
  # lookup embedding weights. One of types.RegularizationType.
  encoder_config.sparse_lookup_regularization = types.RegularizationType.L1

  # The weight used in L1/L2 regularization for the sparse lookup embedding
  # weights.
  encoder_config.sparse_lookup_regularization_weight = 0.00001

  # The choice of regularization for the encoder (fc or residual) weights. One
  # of types.RegularizationType.
  encoder_config.encoder_regularization = types.RegularizationType.L1

  # The weight used in L1/L2 regularization for the encoder weights.
  encoder_config.encoder_regularization_weight = 0.0

  # The weight of the loss for embeddings with a reconstruction loss.
  encoder_config.embedding_loss_weight = 1.0

  # The config for deep embeddings.
  encoder_config.deep = deep_embedding_config(shared)

  # One of types.ModelModes
  encoder_config.mode = shared["mode"]

  return encoder_config


def get_cell_config():
  """Gets the config for a RNN cell."""
  cell_config = configdict.ConfigDict()

  cell_config.leak = 0.001

  return cell_config


def shared_config():
  """Configuration that is needed in more than one field of main config."""
  shared = configdict.ConfigDict()

  # Features from the context of the TF sequence example to use. By default is
  # empty.
  shared.context_features = []

  # Features from the sequence of the TF sequence example to use. By default is
  # set to the default sequence features.
  shared.sequential_features = [
      types.FeatureTypes.PRESENCE_SEQ, types.FeatureTypes.NUMERIC_SEQ,
      types.FeatureTypes.CATEGORY_COUNTS_SEQ
  ]

  # Features for which to just use a lookup embedding.
  shared.identity_lookup_features = [types.FeatureTypes.CATEGORY_COUNTS_SEQ]

  # Number of steps for which event sequence will be unrolled.
  shared.num_unroll = 128

  # The batch size.
  shared.batch_size = 128

  # The mode: one of types.ModelMode
  shared.mode = [types.ModelMode.TRAIN]

  # Each event sequence will chopped into segments of this number of steps.
  # This should be a multiple of num_unroll and is only used in "fast" training
  # mode with pad_sequences=True. RNN states will not be propagated across
  # segment boundaries; as far as the model is concerned, each segment is a
  # complete sequence.
  shared.segment_length = shared.num_unroll * 2

  # Tasks to include in the experiment.
  shared.tasks = (types.TaskNames.ADVERSE_OUTCOME_RISK,
                  types.TaskNames.LAB_REGRESSION)

  return shared


def deep_embedding_config(shared):
  """Options for Deep embeddings."""
  config = configdict.ConfigDict()

  # Embedding activation function.
  # One of: ["tanh", "relu", "lrelu", "swish", "elu", "selu", "elish",
  # "hard_elish", "sigmoid", "hard_sigmoid", "tanh_pen"].
  config.embedding_act = "tanh"

  # Type of encoder to use: types.EmbeddingEncoderType
  config.encoder_type = types.EmbeddingEncoderType.RESIDUAL

  # Encoder layer sizes for presence, numeric and category_counts features in
  # order.
  # Will be mapped to a dict by get_sizes_for_all_features()
  # For FC and residual encoder, the architecture for each feature type is
  # defined by a list specifying the number of units per layer.
  config.encoder_layer_sizes = (2 * [400], 2 * [400], [])

  config.arch_args = configdict.ConfigDict()
  config.arch_args.use_highway_connection = True
  config.arch_args.use_batch_norm = False
  config.arch_args.activate_final = False

  config.tasks = shared.tasks

  # Set configs for SNREncoder
  config.snr = snr_config()

  return config


def get_model_snr_config():
  """SNR related parameters used in the model."""
  config = configdict.ConfigDict()

  # Set parameters for the hard sigmoid
  config.zeta = 3.0
  config.gamma = -1.0
  config.beta = 1.0

  # Set the regularization weight for SNRConnections.
  config.subnetwork_conn_l_reg_factor_weight = 0.0001

  # When set to true it will pass all RNN cell outputs to the tasks.
  # If set to false then it passes only the outputs from the cells on the last
  # layer.
  config.should_pass_all_cell_outputs = True

  # Specify the type of connection between two sub-networks. One of
  # types.SubNettoSubNetConnType.
  config.subnetwork_to_subnetwork_conn_type = types.SubNettoSubNetConnType.BOOL

  # When set to true, it will create a unique routing connection for each input
  # and each task for all RNN cells.
  config.use_task_specific_routing = False

  # Specify how to combine inputs to subnetworks.
  config.input_combination_method = types.SNRInputCombinationType.CONCATENATE

  return config


def snr_config():
  """Options for SNREncoder."""
  config = configdict.ConfigDict()

  # Set parameters for the hard sigmoid
  config.zeta = 3.0
  config.gamma = -1.0
  config.beta = 1.0

  # Set parameters for regularizing the SNREncoder
  config.subnetwork_weight_l_reg_factor_weight = 0.0
  config.subnetwork_weight_l_reg = types.RegularizationType.L2

  config.subnetwork_conn_l_reg_factor_weight = 0.0001

  # Whether to use skip connections in SNREncoder
  config.use_skip_connections = True

  # Whether to use activation before aggregation
  config.activation_before_aggregation = False

  # Specify the type of unit to use in SNREncoder. One of
  # types.SNRBlockConnType
  config.snr_block_conn_type = types.SNRBlockConnType.NONE

  # Specify the type of connection between two sub-networks. One of
  # types.SubNettoSubNetConnType.
  config.subnetwork_to_subnetwork_conn_type = types.SubNettoSubNetConnType.BOOL

  # When set to true, it will create a unique routing connection for each input
  # and each task in all subnetworks of the encoder.
  config.use_task_specific_routing = False

  # Specify how to combine inputs to subnetworks.
  config.input_combination_method = types.SNRInputCombinationType.CONCATENATE

  return config


def get_checkpoint_config():
  """Gets configuration for checkpointing."""
  checkpoint_config = configdict.ConfigDict()

  # Directory for writing checkpoints.
  checkpoint_config.checkpoint_dir = ""

  # How frequently to checkpoint the model (in seconds).
  checkpoint_config.checkpoint_every = 3600  # 1 hour

  # TTL for the training model checkpoints in days.
  checkpoint_config.ttl = 120

  return checkpoint_config
