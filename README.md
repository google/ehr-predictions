# EHR modeling framework
This codebase is intended to accompany the following two papers:
1. Nature Protocols paper titled: Use of deep learning to develop continuous-risk models for adverse event prediction from electronic health records
2. Multitask prediction of organ dysfunction in the intensive care unit using
   sequential subnetwork routing

This codebase illustrates the core components of the continuous prediction model including configurable predictions tasks, interval masking and auxiliary heads. We hope that this will be a useful resource that can be customised by other EHR research labs; however it is not intended as an end-to-end runnable pipeline. Although the full data pre-processing pipeline is not included here as it is highly specific to the patient dataset; we do include synthetic examples of the pre-processing stages with an accompanying data-reading notebook. The codebase also excludes some elements of model evaluation and subgroup analysis; however these are described in the protocol text.

## Using the code
**Prerequisite:** `conda` distribution must be installed on your computer. The
code in this repository was tested with `conda 4.9.1`.

1.  Clone the repository to a local directory and cd into the top level
    directory.
2.  To set up conda run:
    ```
    $ conda clean -y --packages --tarballs
    $ conda env create -f linux-env.yml -p ehr_prediction_modeling/.conda_env
    ```
    to initialize the working environment, then activate the environment as per
    the `conda activate` command displayed in the terminal.
    (e.g. `conda activate /Users/admin/timeseries-informativeness/.conda_env`)
3.  To initialize the ehr_prediction_modeling package run:
    ```
    $ pip install .
    ```
4.  To run training:
    ```
    $ python ehr_prediction_modeling/experiment.py
    ```

## Structure of the code base

* `config.py` defines all the hyperparameters necessary for training a model.
* `configdict.py` defines `ConfigDict`, a "dict-like" data structure with dot access to nested elements.
* `embeddings/` defines all the possible embeddings that map the sparse data to a lower dimensional dense space. `BasicEmbeddingLookup` is a simple sparse matrix multiplication. `DeepEmbedding` is the same thing, followed by extra dense layers.
* `encoder_module_base.py` defines the encoder, a module which passes each of the different feature types through an embedding module and combines the embeddings to get ready for input to a model.
* `experiment.py` is the script to be run to train a model. This version of the code base does not support prediction or evaluation modes yet.
* `mask_manager.py` defines all the masking logic that is used to exclude timesteps from training or evaluation.
* `models/`defines recurrent architectures as subclasses of `sonnet.AbstractModule`. `SequenceModel` is the abstract class from which `BaseRNNModel` inherits, from which `RNNModel` inherits. These classes define the recurrent computational subgraph in Tensorflow. In particular, stacks of RNN layers with highway connections, embedding losses, and hidden state initialization. `ResetCore` is a wrapper around [`sonnet.RNNCore`](https://sonnet.readthedocs.io/en/latest/api.html?highlight=RNNCore#sonnet.RNNCore) that allows resetting the hidden state when a binary flag indicates that a new sequence is starting. `cell_factory.py` allows instantiating the RNN cell based on the value of the config parameter `config.cell_type`. For the time being, only 3 cell types are available. The `SNRNNModel` is the modularized version of the stacked RNN which is used for sub-network routing predictions together with the `SNREncoder` and `SNRTaskLayer`. SNRConnection wrappers as defined in `snr.py` are used throughout for combining the output of a specific part with the learnable routing variables, and then fed as input downstream.
* `tasks/` defines five tasks : mortality, adverse outcome, length of stay, lab values, and readmission. A task is defined by a set of labels and masks. It is in charge of computing the final predictions from the output of the RNN and the loss. When a model is trained to optimize several tasks, the `Coordinator` helps with combining the various training objectives.
* `types.py` defines most constants that are used throughout the code base.
* `utils/` defines various helpers: activation functions, functions to manipulate batches of data, label names, loss functions, and mask names.
* `fake_data/` contains the protocol buffers storing the various stages of processed data. They can be read with `Load_different_processing_stages_of_fake_data.ipynb`.
* `data/` contains utils for reading data using the `tf.data.Dataset` API.


## Data representations
Patients' medical records are represented as [protocol buffers](https://developers.google.com/protocol-buffers). Here we illustrate 3 of the key pre-processing stages: sequentialization, vectorisation, sparse encoding.

### Raw format

See `fake_data/fake_raw_records.pb`.

The raw representation, `FakeRecord`, stores timestamped individual entries of clinical events. Note that the timestamp is actually the age of the patient (`FakeRecord.patient_age`, expressed in days), rather than the actual date of the event. Dates of events and patients’ birth dates are unknown. This is a comparable format to a data dump from an EHR research data warehouse. Clinical events include diagnoses, medications, orders, laboratory tests, etc. For available numerical features, the feature name and feature value are provided as a key, value pair. Categorical features are one hot encoded, converting them into binary features. For illustrative purposes, note that medications are represented here as categorical features rather than using the dose as a continuous feature.

### Sequentialised format

See `fake_data/fake_patients.pb`.

See steps 9 and 10 of the protocol.

The first step in data preprocessing involves grouping these entries by the mock patient ID, and sorting them in time, resulting in a sequential representation: `FakePatient`. This sequential representation gives a sequence of inpatient and outpatient *episodes*, which are themselves sequences of clinical events. There are two types of episodes:  *outpatient events* (for primary care) or *admissions* (for hospital care). The sequence of clinical events within each episode is based on a discretized time, where each step in the sequence corresponds to a fixed-duration time ‘bucket’. Days are sliced into a fixed number of time buckets (e.g. 12am-6am, 6am-12pm, 12pm-6pm, 6pm-12am). Clinical entries that belong to the same time bucket form a single aggregate *clinical event*. Where there are repeated entries for the same feature within a bucket, an aggregated value is kept instead (e.g. median for numerical values or maximum for binary features). Special care is given to entries for which the timestamp has a date but no specific time recorded. To allow the models to process these entries, they have been included in the sequence within a ‘surrogate’ time bucket at the end of each day. In the example above, with 6-hourly time buckets, there would be 5 buckets associated with each day, the 5th being the surrogate. Masking is later applied to ensure inference is not triggered for the surrogate buckets, but this ensures that available data is used while reducing information leakage.

### Vectorised format

See

* `fake_data/vectorized/fake_vectorized_samples.pb`, and
* `fake_data/vectorized/normalized/{normalization}/fake_vectorized_samples.pb`.

See steps 12, 13, and 14 of the protocol.

This representation is transformed into `FakeGenericEventSequence` protocol buffers, which define all the features used for modelling. All numerical observations (e.g. lab and vitals measurements) have 3 features associated with them:

* One continuous feature that stores the observed value. When the value is observed but not known, 0 is used instead.
* Handling missing values is important. They are imputed with 0. To allow the model to make a difference between an absence of measurement and an actual value of 0, we introduce, for each numerical feature, a corresponding binary feature which indicates whether the related numerical feature was measured or not.
* To help models understand which categories individual features belong to, we provide as an optional extension, aggregate counts for how many features from each category there were in that bucket. For example, if there were 7 lab measurements between 12pm and 6pm, the category count for lab tests will be 7. Including these features was not found to boost performance in the published experiments, but we include here as an optional feature.

Discrete features (e.g. diagnoses) have 2 features associated with them: a feature indicating whether it occurred, and a count for all the features of this category.
The same data representation is used to store the normalised numerical values.
See `{normalization}/fake_vectorized_samples.pb`.

### Sparse encoding

See `fake_data/{normalization}/{data_split}.tfrecords`.

See step 14 of the protocol.

Finally, the sequence is transformed into a [`tf.train.SequenceExample`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/SequenceExample), which has a `context` field storing static information like patients' metadata and a `feature_lists` field storing sequential observations. These protocol buffers can be read with the [`tf.data.Dataset` API](https://www.tensorflow.org/tutorials/load_data/tfrecord). Importantly, at this stage the data is in sparse format. Each feature type (numerical, binary, and count) is represented by two sparse tensors ([`tf.sparse.SparseTensor`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/sparse/SparseTensor)). One stores the indices, and the other stores the values of the observed features.
These serialized protos files are named after the 4 available data splits: `train`(~80%), `valid` (~5%), `calib` (5%), and `test` (~10%).

### Feature maps

We provide several feature maps:

* `fake_data/feature_categories_map.pkl`,
* `fake_data/numerical_map.pkl`,
* `fake_data/presence_map.pkl`, and
* `fake_data/features_ids_per_category_map.pkl`

The first three files are serialized dictionaries that can be read with `pickle.load`. They are mappings between feature names / ids (see `FakePatient` and `FakeRecord` protos) and feature indices (the column identifier in the input sparse matrix).

The last is a mapping between feature categories and lists of features ids belonging to this category.

## Synthetic data
**For illustration purposes**, we include a random sample of 1000 fake patient records that were generated in raw `FakeRecord` format and transformed by a proprietary data processing pipeline, the outputs of which are available in `fake_data/`.

* Values and events are randomly sampled and do not reflect any clinical reality nor correspond to any actual patient. This data is meant merely as illustration, as it does not have the same properties as an actual EHR data would in terms of feature sparsity or distribution.
* All clinical concepts and demographic attributes have been replaced with generic names.
* Numerical values are sampled following Uniform distributions.
* 10 feature categories are included, 8 of them taking binary values, and 2 continuous values.
* One generic adverse clinical outcome is included. `AdverseOutcome` can occur several times during an admission and typically lasts several time steps.

## Task framework
See step 19 of the protocol.

Neural networks can be trained by optimising flexible training objectives. In particular, several competing scalar functions of the weights can be combined and simultaneously minimised via stochastic gradient descent. A neural network is trained to improve its predictive performance on one or more tasks. A *task* defines a supervised training objective and evaluation metrics. In particular, it specifies what the targets are, how and when the objective / metrics should be computed. Some time steps are to be excluded from training and/or evaluation. This is handled via boolean masks, that are specified in the task. A *task coordinator* manages and combines the different objectives.
The included suite of tasks consists of:

* `AdverseOutcome`: this task is a classification problem aimed at predicting whether a generic adverse outcome will happen over the next X hours. This event may occur multiple times during an admission, or not at all. When happening, it typically lasts for one or more consecutive time steps. See `adverse_outcome_task.py`.
* `MortalityRisk`: this task is a classification problem aimed at predicting whether a patient will die over the next X hours (days are typically a more suited timescale for these predictions, but the time windows are expressed in hours for consistency with the other tasks). See `mortality_task.py`.
* `LengthOfStay`: this task can be both a classification (will the admission stay be longer than X days ?) or regression problem (how many days are left in admission?). See `los_task.py`.
* `Readmission`: this task is a classification problem aimed at predicting whether a patient will be readmitted in the X days following their discharge. All time steps that are not a discharge event are masked out from the training objective and performance metrics. See `readmission_task.py`.
* `LabsRegression`: this task is a regression problem aimed at predicting what the maximum value of a set of labs will be over the following X hours. Only the maximum aggregation (rather than other techniques like minimum, mean, median, standard deviation...) is included here for simplicity, but should be customised to the labs of interest. See `labs_task.py`.

## Labels

The task framework is associated with several labels for each task.

### AdverseOutcome
`adverse_outcome_within_X_hours` for `X` in `[6, 12, 24, 36, 48, 72]`: These labels indicate whether this adverse event occurs at least once in the provided time window. The values that these labels take are integers between 0 and 4 included, 0 meaning unknown, and 1 to 4 describing different severity levels. See `adverse_outcome_task.AdverseOutcomeLevels`.

### Mortality
`mortality_within_X_hours` for `X` in `[24, 168, 720]`: These labels indicate whether the patient will die during the following X hours. The possible values are 0 and 1.

`mortality_in_admission`: This label is 1 if the patient dies in their remaining hospital admission, and 0 otherwise. This is an alternative formulation of the task that is configured via the task config keyword `mortality_in_admission=True/False`.

### LengthOfStay
`length_of_stay`: This numerical value represents how many days are left in the admission.

### Readmission
`time_until_next_admission`: Denotes the number of hours between the current time bucket and the next admission. It can be `+infinity` if there are no further admissions for that patient.

`discharge_label`: This is 1 only at the timestep of discharge (0 in cases of mortality or hospital transfer). This is used to define `AT_DISCHARGE_MASK`, so that the model is trained / evaluated only at time of discharge for eligible patients.

### LabsRegression
`lab_<lab_id>_in_Xh`: the aggregated (e.g. max, median, std, ...) value of the given lab over the relevant time window. For the sake of simplicity, we only include one type of aggregation in this dataset.

## Masking
See step 16 of the protocol.

Boolean masks are used to exclude specific time steps from training or evaluation. All masks are defined in `mask_utils.py`. Composite masks are obtained by multiplying individual component masks (so that a time step is masked if any of the component masks requires it to be so, see `mask_utils.get_combined_mask`). Some component masks are common to all tasks, while others are specific to one task in particular. For example:

* `IGNORE_MASK`: all outpatient events are masked out from both training and evaluation, because the model is trained to predict inpatient adverse outcomes. Outpatient events are still included in the medical history so that the model can process the entire patient history. Surrogate buckets (events for which time of day is unknown and that are moved to the end of the day) are also masked out by this mask.
* `PADDED_EVENT_MASK`: Since time series have different durations, they are padded with zeros to allow batching. These time steps are excluded from training and evaluation, since they do not correspond to any actual event. 
* `UNKNOWN_LOOKAHEAD_MASK`: When predicting continuous numerical values, all the time steps with unknown values are masked out. When the severity of a specific adverse event is not known (label 0), then this timestep is masked, too.
* `INTERVAL_MASK`: The prediction task may not be clinically actionable during some intervals of the patient admission - e.g. predicting AKI during a period of dialysis treatment. These periods can be censored with interval masks.
* `PATIENT_MASK`: you may want to filter out entire patients’ medical histories from training, for example when they do not meet inclusion criteria. This mask allows on-the-fly removal of entire patients’ histories during training or evaluation.
* `AROUND_ADMISSION_MASK`: Note that masking can also be used for specifying the temporal formulation of the problem. Predictions can be continuous (i.e. one prediction at each timestep) or static (only one prediction per admission, by masking out all but one time step). For example, predicting mortality 12h/24h after admission is a common benchmark in the literature.


## Modelling
See step 19 of the protocol.

### Stacked RNN Model
The model consists of an Multilayer Perceptron (MLP) that projects sparse inputs at all time steps to a lower dimensional dense space (step 18 of the protocol). The time series formed by these projections is then fed to a deep RNN with highway connections that predicts one or more targets, for one or more tasks. The combined training objective is a weighted sum of the individual tasks’ training objectives. They are combined by the task Coordinator. For binary classification tasks, the cross entropy loss is a proxy for the negative loglikelihood, while the L2 loss is a proxy for the Brier score. Regression tasks can use L1 or L2 losses. See `utils/loss_utils.py`. Weights are updated via stochastic gradient descent for a given number of steps.

### SNRNN
Similar to the Stacked RNN Model, SNRNN is a deep RNN that predicts one or more targets, for one or more tasks. The difference is that each layer is split up into multiple cells, whose outputs are connected to the following layer through routing connections. More information and exact diagrams of the setup can be found [here](https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocab101/6307184).

## Data reading
Data is read in [`tf.train.SequenceExample`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/SequenceExample) format using the [`tf.data.Dataset` API](https://www.tensorflow.org/tutorials/load_data/tfrecord). Since time series have different lengths, they are padded so that they can be batched together. All batches are in time-major format (i.e. their dimension is `[num_unroll, batch_size, num_features]`). Padded time steps are discarded in the training loss using a boolean mask. A binary flag indicates when the hidden state of the RNN should be reset (when a new patient’s history starts).

See `tf_dataset.py` and `tf_dataset_utils.py`.

### Segmentation of long sequences
Due to the long-tailed nature of medical record data, some patient histories can be much longer than the mean/median. Naive padding of the resulting sequential data would result in poor predictive performance, as many training batches will be dominated by a very small number of sequences.

To address this problem, we split each [`tf.train.SequenceExample`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/train/SequenceExample) in time by a multiple of the unroll length, e.g. if we unroll the RNN by 128 steps during training then we might choose to divide the training sequences into segments of 256 steps. These `segments` are then shuffled and batched as though they are sequences from different patients. The resulting training batches are more densely packed and rarely dominated by data from a single patient.


