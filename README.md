# EvoDENSS

[![](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/) [![](https://img.shields.io/badge/PyTorch-2.0.0-blue.svg)](https://pytorch.org/get-started/previous-versions/) [![](https://img.shields.io/badge/cudatoolkit-11.3-blue.svg)](https://developer.nvidia.com/cuda-downloads/)

[![](https://img.shields.io/badge/License-Apache_2.0-green.svg)]()

<!---
![t](https://img.shields.io/badge/status-maintained-green.svg)
[![](https://img.shields.io/github/license/adrianovinhas/fast-denser-adriano.svg)](https://github.com/adrianovinhas/fast-denser-adriano/blob/master/LICENSE.md)
-->

ENERGIZE stands for **E**nergy-efficient **N**euro**E**volution fo**R** **G**eneral**IZE**d learning.

It is an adaptation of the [EvoDENSS](https://github.com/adrianovinhas/evodenss/) framework, itself drawing inspiration from [Fast-DENSER](https://github.com/fillassuncao/fast-denser3).

## Installing

In order to run ENERGIZE, one needs to install the relevant dependencies. There are two ways to install the framework:

##### 1. Conda

A conda environment can be created from an exported yml file that contains all the required dependences:

```
conda env create -f environment.yml
```

After the environment is created, just activate it in order to be able to run your code:

```
conda activate energize
```

##### 2. pip

Alternatively, you can use the `requirements.txt` file, but you will be on your own to install cudatoolkit and other libraries that might be required to enable GPU acceleration.

```
pip install -r requirements.txt
```

**Note:** Installing ENERGIZE as a Python library is not yet supported

## Running ENERGIZE

- In order to run fast ENERGIZE, you need to run the evodenss module as a script:

```
python3 -u -m energize.main \
    -d <dataset_name> \
    -c <config_path> \
    -g <grammar_path> \
    -r <#run>
```

Example:

```
python3 -u -m energize.main \
    -d mnist \
    -c example/example_config.yaml \
    -g example/example.grammar \
    --run 0 \
    --gpu-enabled
```

In case several seeds are needed to be run, that can be done with Bash:

```
for i in {7..9}; do \
    python3 -u -m energize.main \
    -d mnist \
    -c example/example_config.yaml \
    -g example/example.grammar \
    -r $i \
    --gpu-enabled; \
done
```

Externally to the code itself, two main files are required to execute any run.

1. A grammar that shapes the search space by setting the possibilities within each macro block.
2. A configuration file that sets miscellaneous parameters that affect the outcome of the evolutionary run. These can se related with the evolutionary process itself, or the networks that are generated.

## Testing

Unit tests can be executed via `pytest`:

```
pytest tests
```

In case one wants to do it with coverage report:

```
coverage run --source energize -m pytest -v tests
coverage report
```

#### Command-line flags

- `-c`/`--config-path`: Sets the path to the config file to be used;
- `-d`/`--dataset-name`: Name of the dataset to be used. At the moment, `mnist`, `fashion-mnist`, `cifar10` and `cifar100` are supported.
- `-g`/`--grammar-path`: Sets the path to the grammar to be used;
- `-r`/`--run`: Identifies the run id and seed to be used;
- `--gpu-enabled`: When used, it enables GPU processing.

#### ENERGIZE features

###### 1. ...
