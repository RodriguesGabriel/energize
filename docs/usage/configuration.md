---
layout:
  title:
    visible: true
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# Configuration

The configuration file can be either a JSON or a YAML file. We will present documentation for the JSON file. The YAML is similar.

The configuration file can be composed of up to 5 blocks:

* `checkpoints_path` (_str_) Specifies the path to the directory where checkpoints and experiment results will be saved.     &#x20;
* `statistics_format` (_str_) Determines the format for saving statistics. It can be either "json" or "csv".
* [`energize`](configuration.md#energize) (_dict_) Options related to the ENERGIZE main features.
* [`evolutionary`](configuration.md#evolutionary) (_dict_) Options for the evolutionary algorithm.
* [`network`](configuration.md#network) (_dict_) Options concerning the architectures and learning parameters of the networks.

***

## Energize

The `energize` section of the configuration file contains options specific to the ENERGIZE framework. If this section is absent in the configuration file, the framework is equivalent to the Fast-DENSER framework.

#### `measure_power` (_dict_)

Specifies whether the power usage should be measured or not as explained in [power-measurement.md](../features/power-measurement.md "mention")

* **train**: _(bool)_ Enables power measurement during training.
* **test**: _(bool)_ Enables power measurement during testing.
* **num\_measurements\_test**: _(int)_ Specifies the number of power measurements to take during testing.
* **modules**: _(bool)_ Enables measurement of power consumption per module as specified in [module-reuse.md](../features/module-reuse.md "mention")

#### `model_partition` (_bool_) Enables model splitting

#### `model_partition_n` (_int_) Specifies the number of models to create from the full model (not tested for $$n  \ne 2$$).

#### `ensure_gpu_exclusivity` (_bool_) Ensures exclusive GPU usage through the usage of the _Compute Exclusive_ mode of the NVIDIA GPU, as specified in [power-measurement.md](../features/power-measurement.md "mention")

***

## Evolutionary

This section of the configuration file contains parameters that drive the evolutionary algorithm within ENERGIZE.&#x20;

**`generations`** _(int)_: Specifies the number of generations.

**`lambda`** _(int)_: Specifies the $$\lambda$$ parameter, i.e., the number of offspring generated in each generation from the selected parent.

**`max_epochs`** _(int)_: Specifies the maximum cumulative number of epochs for training all individuals.

**`mutation`** _(dict)_: Defines mutation probabilities for various operations, namely:

* `add_connection` _(float)_: Probability of adding a connection between layers.
* `remove_connection` _(float)_: Probability of removing a connection between layers.
* `add_layer` _(float)_: Probability of adding a new layer.
* `reuse_layer` _(float)_: Probability of reusing an existing layer.
* `remove_layer` _(float)_: Probability of removing a layer.
* `reuse_module` _(float)_: Probability of reusing an existing module.
* `remove_module` _(float)_: Probability of removing a module.
* `dsge_layer` _(float)_: Probability of performing a DSGE mutation on a layer.
* `macro_layer` _(float)_: Probability of applying a macro mutation on a layer.
* `train_longer` _(float)_: Probability of allowing the training of a model for a longer duration.

**`track_mutations`** _(bool)_: Specifies whether to track mutations. If activated, each mutation is registered in the statistics output files.&#x20;

**`fitness_metric`** (str_)_: Defines the fitness metric for evaluating individuals. Note that only one  `fitness_metric` or `fitness_function` can be used at a time. The allowed values are specified below.

**`fitness_function`** _(list\[dict])_: Defines the fitness function for evaluating individuals. Note that only one  `fitness_metric` or `fitness_function` can be used at a time. Each entry in the `fitness_function` list is composed by the following parameters:

* `metric` _(str)_: Specifies the fitness metric to be used for evaluation. The allowed values are specified below.
* `objective` _(str)_: Defines whether the objective is to `maximize` or `minimize` the specified metric.
* `weight` _(float)_: Assigns a weight to the metric, influencing its importance in the overall fitness evaluation.
* `conditions` _(list\[dict])_ (optional): Allows the definition of additional conditions for the fitness function. Each element of the list is considered as a conjunction, meaning they are combined using the "and" operator. Each element in the list represents a condition specified as a dictionary with the following properties:
  * `metric` _(str)_: Specifies the fitness metric for the condition.
  * `type` _(str)_: Specifies whether the condition is "less\_than" or "greater\_than".
  * `value` _(float)_: Specifies the threshold value for the condition.
  * For example, specifying that "accuracy\_0" must be "greater\_than" the value 0.5 means that the metric in question will only be considered if "accuracy\_0" is greater than 0.5

{% hint style="info" %}
**Fitness metrics:**&#x20;

* **accuracy**: measures the model's predictive performance by calculating the proportion of correctly classified instances.
* **loss:** quantifies the model's error during training, with lower values indicating better performance.
* **power**: power usage of the GPU during inference (in W).
* **energy**: energy consumed by the GPU during inference (in kJ).

If `model_partition` is activated, then the fitness metric should be appended with `_i` with `i` being the ID of the model (e.g., accuracy\_0 is the accuracy of the full model).
{% endhint %}

***

## Network

`architecture` _(dict)_: Specifies the architecture configuration.

* `reuse_layer` _(float)_: Specifies the probability of reusing an existing layer during network architecture initialization.
* `macro_structure` (_list\[str])_: Specifies the macrostructure symbols to use. Depends on the grammar.
* `output` _(str)_: Specifies the type of output layer, such as "softmax".
* `modules` _(list\[dict])_: Specifies the allowed configuration of individual modules.
  * `name` _(str)_: Specifies the name of the module.
  * `network_structure_init` _(list\[int])_: Specifies the possibilities for the number of layers of the module when initialized, i.e., the number of layers will be randomly chosen from the values of this list.
  * `network_structure` _(list\[int])_: Specifies the minimum and maximum number of layers of the networks. Format: \[min, max].
  * `levels_back` _(int)_: Specifies the maximum number of levels back to connect to when initializing the module. For example, a value of 2 levels back allows to connect to the previous two layers. This is also considered in the `add_connection` mutation.

`learning` _(dict)_: Specifies the learning configuration.

* `data_splits` _(dict)_: Specifies the data splits for training, validation, and testing datasets.
  * `evo_train` (_float_ $$\in [0, 1]$$)
  * `evo_validation` (_float_ $$\in [0, 1]$$)
  * `evo_test` (_float_ $$\in [0, 1]$$)
* `learning_type` _(str)_: Specifies the type of learning. For now, the only option is "supervised".
* `default_train_time` _(int)_: Specifies the default training time in seconds allocated by default to each individual.
* `augmentation` _(dict)_: Specifies data augmentation configurations for training and testing datasets.
  * `train` _(dict)_: Specifies augmentation configurations for the training dataset.
  * `test` _(dict)_: Specifies augmentation configurations for the testing dataset.
  * In both cases, the options are:
    * TBW (you can consult the file [energize/networks/torch/transformers.py](../../energize/networks/torch/transformers.py) for more information)
