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
"""Labs task implementation to be used in experiments."""

from typing import Dict, List, Mapping, Optional, Tuple, Union

from ehr_prediction_modeling import types
from ehr_prediction_modeling.tasks import base_task
from ehr_prediction_modeling.tasks import task_data
from ehr_prediction_modeling.tasks import task_masks
from ehr_prediction_modeling.utils import batches
from ehr_prediction_modeling.utils import label_utils
from ehr_prediction_modeling.utils import loss_utils
from ehr_prediction_modeling.utils import mask_utils
import tensorflow.compat.v1 as tf

from ehr_prediction_modeling import configdict


class LabsRegression(base_task.Task):
  """Task implementation for prediction of future lab or vitals value."""

  task_type = types.TaskTypes.LAB_REGRESSION

  # Only max is illustrated in the fake dataset.
  _aggregate_types = ("max", "min", "mean", "std")

  @property
  def default_masks(self) -> List[str]:
    return [mask_utils.IGNORE_MASK, mask_utils.UNKNOWN_LOOKAHEAD_MASK]

  @property
  def _unformatted_supported_eval_masks(self) -> Dict[str, List[str]]:
    """See base class."""
    return {
        task_masks.Eval.BASE:
            self.default_masks,
    }

  def __init__(self, config: configdict.ConfigDict):
    label_keys = []
    self._target_names = []
    self._num_labs = len(config.labs)
    self._accumulate_logits = config.accumulate_logits
    self._window_times = []  # one value per (lab, aggregation, time horizon)
    for lab_id, lab_name in config.labs:
      for time_window in config.window_times:
        for aggregation in config.aggregations:
          if aggregation not in LabsRegression._aggregate_types:
            raise ValueError(
                "LabsRegression aggregation {} is invalid. Must be one of {}"
                .format(aggregation, LabsRegression._aggregate_types))
          label_keys.append(
              label_utils.get_lab_label_lookahead_key(
                  lab_id,
                  time_window,
                  # In the fake data, max aggregation has no suffix.
                  suffix=None if aggregation == "max" else aggregation))
          lab_name = lab_name.replace(" ", "_").lower()
          target_name = "{aggregation}_{lab_name}_in_{hours}h".format(
              aggregation=aggregation, lab_name=lab_name, hours=time_window)
          self._window_times.append(time_window)
          self._target_names.append(target_name)
    super().__init__(config, label_keys=label_keys)


  def get_label_dicts(
      self
  ) -> Tuple[Dict[Optional[str], Union[
      tf.FixedLenSequenceFeature, tf.FixedLenFeature]], Dict[Union[
          str, None], Union[tf.FixedLenSequenceFeature, tf.FixedLenFeature]]]:
    """Gets the feature dictionaries to parse a tf.SequenceExample.

    These dicts are used in the parse op (either batched or non-batched):
    https://www.tensorflow.org/api_docs/python/tf/io/parse_single_sequence_example#args
    They should include any feature that is used by this task.

    Returns:
      context_d: Dictionaries of TF features to read in context.
      sequence_d: Dictionaries of TF features to read in sequence.
    """
    context_d = {}
    sequence_d = {
        label_key: tf.FixedLenSequenceFeature([1], tf.float32)
        for label_key in self._label_keys
    }
    return context_d, sequence_d

  @property
  def num_targets(self) -> int:
    return len(self._label_keys)

  @property
  def target_names(self) -> List[str]:
    return self._target_names

  @property
  def window_hours(self) -> List[int]:
    return self._window_times

  @property
  def prediction_task_type(self) -> str:
    return types.TaskType.REGRESSION

  def _accumulate_per_lab(self, logits: tf.Tensor) -> tf.Tensor:
    """Accumulates logits across time windows for each lab separately."""
    logits_split_by_lab = tf.split(
        logits, num_or_size_splits=self._num_labs, axis=-1)
    accumulated_logits_list = [
        tf.cumsum(lab_logits, axis=-1) for lab_logits in logits_split_by_lab
    ]
    return tf.concat(accumulated_logits_list, axis=-1)

  def _get_all_task_variables(
      self, batch: batches.TFBatch,
      model_output: tf.Tensor) -> task_data.TaskVariables:
    """Computes variables for LabsRegression task.

    Args:
      batch: tf.NextQueuedSequenceBatch, containing a batch of data.
      model_output: Tensor, the output from the model, shape wnt [num_unroll,
        batch_size, dim_model_output].

    Returns:
      task_data.TaskVariables with all the variables from this task.
    """
    logits = self.layer.get_logits(model_output)

    if self._accumulate_logits:
      logits = self._accumulate_per_lab(logits)

    targets = self.get_targets(batch)

    train_loss_mask = self.get_train_mask(batch)
    loss = loss_utils.loss_fn(logits, targets, train_loss_mask,
                              self._config.loss_type)

    eval_mask_dict = self.get_eval_mask_dict(batch)

    return task_data.TaskVariables(
        loss=loss,
        targets=targets,
        predictions=logits,
        train_mask=train_loss_mask,
        eval_mask_dict=eval_mask_dict,
    )

  @classmethod
  def config(
      cls,
      window_times: List[int],
      aggregations: List[str],
      labs: Optional[List[Tuple[str, str]]] = None,
      train_mask: str = task_masks.Train.BASE,
      eval_masks: Optional[List[str]] = None,
      loss_type: str = types.TaskLossType.L2,
      loss_weight: float = 5.0,
      accumulate_logits: bool = False,
      task_layer_type: str = types.TaskLayerTypes.MLP,
      task_layer_sizes: Optional[List[int]] = None,
      regularization_type: str = types.RegularizationType.NONE,
      regularization_weight: float = 0.,
      name: str = types.TaskNames.LAB_REGRESSION,
      snr_config: Optional[configdict.ConfigDict] = None,
  ) -> configdict.ConfigDict:
    """Generates a config object for LabsRegression.

    Args:
      window_times: list of int, prediction windows for the labs regression.
      aggregations: list of string, aggregations to use per lab. Should be one
        LabsRegression._aggregate_types.
      labs: list of tuples, (lab_id, lab_name). If not given, a default
        list will be used.
      train_mask: str, name of the mask to be used in train.
      eval_masks: list of str, names of the masks to be used in eval.
      loss_type: str, type of loss to be used.
      loss_weight: float, weight of this task loss.
      accumulate_logits: bool, whether to create a CDF over logits for each lab
        task to encourage monotonicity for increasing time windows. Should only
        be imposed if we are predicting the maximum lab value.
      task_layer_type: one of types.TaskLayerTypes - the type of layer to use.
      task_layer_sizes: array of int, the size of the task-specific layers to
        pass the model output through before a final logistic layer. If None,
        there is just the final logistic layer.
      regularization_type: one of types.RegularizationType, the regularization
        to be applied to the task layer(s).
      regularization_weight: float, the weight of the regularization penalty to
        apply to logistic layers associated with this task.
      name: str, name of this task for visualization and debuggigng.
      snr_config: configdict.ConfigDict, containing task layer sub-network
        routing parameters.

    Returns:
      A ConfigDict to be used to instantiate a LabsRegression task.
    """
    config = configdict.ConfigDict()
    config.task_type = LabsRegression.task_type
    config.labs = labs or [("42", "Lab 1"), ("43", "Lab 2"),
                           ("44", "Lab 3"), ("45", "Lab 4"),
                           ("46", "Lab 5")]
    config.aggregations = aggregations
    config.time_since_event_hours_list = []  # unused, here for consistency
    config.window_times = sorted(window_times)
    config.train_mask = train_mask
    config.eval_masks = eval_masks or []
    config.loss_type = loss_type
    config.loss_weight = loss_weight
    config.accumulate_logits = accumulate_logits
    config.name = name
    config.task_layer_sizes = task_layer_sizes or []
    config.regularization_type = regularization_type
    config.regularization_weight = regularization_weight
    return config

  @classmethod
  def default_configs(cls) -> List[configdict.ConfigDict]:
    """Generates a default config object for LabsRegression."""
    return [
        LabsRegression.config(
            window_times=label_utils.DEFAULT_LOOKAHEAD_WINDOWS,
            aggregations=["max"],
            eval_masks=None,
            loss_type=types.TaskLossType.L2,
            loss_weight=5.0)
    ]
