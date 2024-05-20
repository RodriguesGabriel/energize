# Basic usage

In order to run fast ENERGIZE, you need to run the energize module as a script:

```
python3 -u -m energize.main \
    -d <dataset_name> \
    -c <config_path> \
    -g <grammar_path> \
    -r <#run> \
    --gpu-enabled
```

Example:

```
python3 -u -m energize.main \
    -d mnist \
    -c example/example_config.json \
    -g example/energize.grammar \
    -r 0 \
    --gpu-enabled
```

In case several seeds are needed to be run, that can be done with Bash:

```
bash batch.sh example/example_config.json example/energize.grammar fashion-mnist 5
```

Externally to the code itself, two main files are required to execute any run.

1. A grammar that shapes the search space by setting the possibilities within each macro block.
2. A configuration file that sets miscellaneous parameters that affect the outcome of the evolutionary run. These can se related with the evolutionary process itself, or the networks that are generated.

## Command-line flags

* `-c`/`--config-path`: Sets the path to the config file to be used;
* `-d`/`--dataset-name`: Name of the dataset to be used. At the moment, `mnist`, `fashion-mnist`, `cifar10` and `cifar100` are supported.
* `-g`/`--grammar-path`: Sets the path to the grammar to be used;
* `-r`/`--run`: Identifies the run id and seed to be used;
* `--gpu-enabled`: Enables GPU processing.
