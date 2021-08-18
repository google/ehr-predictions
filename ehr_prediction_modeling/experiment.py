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
"""Experiment runner."""
import os

from absl import app
from absl import logging
from ehr_prediction_modeling import config as experiment_config
from ehr_prediction_modeling import embeddings
from ehr_prediction_modeling import encoder_module_base
from ehr_prediction_modeling import losses
from ehr_prediction_modeling import types
from ehr_prediction_modeling.data import tf_dataset
from ehr_prediction_modeling.eval import metrics_coordinator as metrics
from ehr_prediction_modeling.models import model_utils
from ehr_prediction_modeling.models import rnn_model
from ehr_prediction_modeling.tasks import coordinator
import tensorflow.compat.v1 as tf


def get_checkpoint_dir(config, mode):
  ttl = "ttl=%sd" % config.ttl
  return os.path.join(config.checkpoint_dir, "checkpoints", ttl, mode)


def get_task_from_config(config, task_name):
  """Returns an instantiated Task based on the provided task name."""
  if task_name not in config.task_configs:
    raise ValueError(
        "Task %s is not present in the list of task configurations: %s." %
        (task_name, config.task_configs.keys()))
  task_config = config.task_configs[task_name]

  if task_config.task_type not in experiment_config.TASK_MAPPING:
    raise ValueError("config.tasks.type unknown: %s" % task_config.task_type)

  task = experiment_config.TASK_MAPPING[task_config.task_type](task_config)
  return task


def get_task_coordinator(config):
  task_list = [
      get_task_from_config(config, task_name) for task_name in config.tasks
  ]
  return coordinator.Coordinator(
      task_list, optimizer_config=config.get("optimizer"))


def eval_fn_on_data_split(task_coordinator, metrics_coordinator, session,
                          eval_batch_gen, eval_task_vars, split_name,
                          current_step):
  """Runs evaluation of a given datasplit and logs metrics."""
  # The dataset needs to be re-initialized for each epoch since we iterate the
  # entire data split.
  session.run(eval_batch_gen.iterator.initializer)
  task_prediction_types = task_coordinator.task_prediction_types
  target_names_list = task_coordinator.target_names_list
  fetches = {
      "task_variables_list": eval_task_vars,
  }

  batch_count = 0
  while True:
    try:
      fetches_np = session.run(fetches)
      for (target_names, task_type,
           task_variables) in zip(target_names_list, task_prediction_types,
                                  fetches_np["task_variables_list"]):
        if task_type == types.TaskType.BINARY_CLASSIFICATION:
          metrics.add_batch_to_binary_metrics_data(
              metrics_coordinator=metrics_coordinator,
              target_names=target_names,
              predictions=task_variables.predictions,
              binary_targets=task_variables.targets,
              eval_mask_dict=task_variables.eval_mask_dict,
              split_name=split_name)

        elif task_type == types.TaskType.REGRESSION:
          metrics.add_batch_to_regression_metrics_data(
              metrics_coordinator=metrics_coordinator,
              target_names=target_names,
              predictions=task_variables.predictions,
              targets=task_variables.targets,
              eval_mask_dict=task_variables.eval_mask_dict,
              split_name=split_name)

        else:
          raise ValueError("Unsupported task type for evaluation: %s" %
                           task_type)

    except tf.errors.OutOfRangeError:
      # OutOfRangeError is the normal error thrown when the queue is empty
      # due to the epoch limitation.
      break
    batch_count += 1

  logging.info("Evaluated %s batches.", batch_count)
  metrics_coordinator.log_metrics(current_step, clear_data=True)


def setup_eval(config, task_coordinator, split, model, encoder):
  batch_gen = tf_dataset.BatchGenerator(config, False, task_coordinator, split)
  batch = batch_gen.batch
  features, time_vect = encoder.embed_batch(batch)
  forward_return = model(features, batch.is_beginning_sequence, time_vect)
  tasks_graph = task_coordinator.get_coordinator_variables(
      batch, forward_return.model_output)
  return (batch_gen, tasks_graph.task_variables_list)


def run(config):
  """Build model and runs experiment."""
  task_coordinator = get_task_coordinator(config)

  tf.random.set_random_seed(config.get("seed", 0))
  logging.info(config)

  metrics_coordinator = metrics.MetricsCoordinator()

  embedding_classes = {
      types.EmbeddingType.LOOKUP: embeddings.BasicEmbeddingLookup,
      types.EmbeddingType.DEEP_EMBEDDING: embeddings.DeepEmbedding
  }
  encoder = encoder_module_base.EncoderModule(config.encoder, embedding_classes)

  model_init_kwargs = {
      "config": config.model,
      "embedding_size": encoder.get_total_embedding_size()
  }
  base_model = rnn_model.RNNModel(**model_init_kwargs)
  model = model_utils.RNNModelWithPersistentState(base_model)
  optimizer = model_utils.get_optimizer_from_config(config.optimizer)

  batch_gen = tf_dataset.BatchGenerator(config, True, task_coordinator, "train")
  batch = batch_gen.batch
  features, time_vect = encoder.embed_batch(batch)
  forward_return = model(features, batch.is_beginning_sequence, time_vect)
  tasks_graph = task_coordinator.get_coordinator_variables(
      batch, forward_return.model_output)
  embedding_loss, _ = encoder.get_embedding_loss(batch)

  loss = tasks_graph.combined_loss
  loss += encoder.get_embedding_regularization_loss()
  loss += embedding_loss
  loss += model.get_model_regularization_loss()

  losses_per_task = {}
  for task_name, task_vars in zip(task_coordinator.task_names,
                                  tasks_graph.task_variables_list):
    losses_per_task[task_name] = task_vars.loss

  loss += task_coordinator.get_task_regularization_losses()

  loss_to_vars = losses.get_loss_to_variables_dict(
      model=model,
      encoder=encoder,
      losses_per_task=losses_per_task,
      all_variables=tf.trainable_variables(),
      total_loss=loss)
  step = model_utils.multiple_loss_optim_fn(
      optimizer, loss_to_vars, norm_clip=config.optimizer.norm_clip)

  split = config.splits_to_evaluate
  eval_batch_gen, eval_task_vars = setup_eval(config, task_coordinator, split,
                                              model, encoder)

  with tf.control_dependencies([step]):
    scalar_loss = tf.reduce_mean(loss)
    step_cnt = tf.train.get_or_create_global_step()
  current_step = 0

  checkpoint_dir = get_checkpoint_dir(config.checkpoint, "train")
  with tf.train.MonitoredTrainingSession(
      is_chief=True,
      checkpoint_dir=checkpoint_dir,
      save_checkpoint_secs=config.checkpoint.checkpoint_every,
      save_summaries_steps=None,
      save_summaries_secs=None,
      config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False)
  ) as session:
    fetches = {
        "step": step_cnt,
        "loss": scalar_loss,
    }
    while current_step < config.model.num_steps:
      fetches_np = session.run(fetches)
      current_step = fetches_np["step"]
      if current_step % 100 == 0:
        logging.info("step %s, fetches: %s", current_step, fetches_np)
        logging.info("Starting evaluation on data split: %s", split)
        eval_fn_on_data_split(task_coordinator, metrics_coordinator, session,
                              eval_batch_gen, eval_task_vars, split,
                              current_step)


def main(_):
  config = experiment_config.get_config()
  run(config)


if __name__ == "__main__":
  app.run(main)
