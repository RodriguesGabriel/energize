# ENERGIZE
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11220573.svg)](https://doi.org/10.5281/zenodo.11220573) [![](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/) [![](https://img.shields.io/badge/PyTorch-2.2.0-blue.svg)](https://pytorch.org/get-started/previous-versions/) [![](https://img.shields.io/badge/cudatoolkit-12.1.0-blue.svg)](https://developer.nvidia.com/cuda-downloads/)

[![](https://img.shields.io/badge/License-Apache_2.0-green.svg)]()

<!---
![t](https://img.shields.io/badge/status-maintained-green.svg)
[![](https://img.shields.io/github/license/adrianovinhas/fast-denser-adriano.svg)](https://github.com/adrianovinhas/fast-denser-adriano/blob/master/LICENSE.md)
-->

ENERGIZE stands for **E**nergy-efficient **N**euro**E**volution fo**R** **G**eneral**IZE**d learning.

It is an adaptation of the [EvoDENSS](https://github.com/adrianovinhas/evodenss/) framework, itself drawing inspiration from [Fast-DENSER](https://github.com/fillassuncao/fast-denser3).


## Features

* feature1
* feature2


## Installation

In order to run ENERGIZE, one needs to install the relevant dependencies. There are two ways to install the framework:

### Conda

A conda environment can be created from an exported yml file that contains all the required dependences:

```
conda env create -f environment.yml
```

After the environment is created, just activate it in order to be able to run your code:

```
conda activate energize
```

### pip

Alternatively, you can use the `requirements.txt` file, but you will be on your own to install cudatoolkit and other libraries that might be required to enable GPU acceleration.

```
pip install -r requirements.txt
```

**Note:** Installing ENERGIZE as a Python library is not yet supported

## Getting Started

Example:

```
python3 -u -m energize.main \
    -d mnist \
    -c example/example_config.json \
    -g example/energize.grammar \
    --run 0 \
    --gpu-enabled
```

## Command-line flags

- `-c`/`--config-path`: Sets the path to the config file to be used;
- `-d`/`--dataset-name`: Name of the dataset to be used. At the moment, `mnist`, `fashion-mnist`, `cifar10` and `cifar100` are supported.
- `-g`/`--grammar-path`: Sets the path to the grammar to be used;
- `-r`/`--run`: Identifies the run id and seed to be used;
- `--gpu-enabled`: When used, it enables GPU processing.


## Documentation
Please visit our [GitBook Documentation](https://bai-cisuc.gitbook.io/energize)

<!-- ## Contributing -->

<!-- ## License -->

## Contact

- Gabriel Cortês (cortes@dei.uc.pt)
- Nuno Lourenço (naml@dei.uc.pt)
- Penousal Machado (machado@dei.uc.pt)


## Publications
The methods used in these frameworks are mainly described in the following [paper](https://arxiv.org/abs/2401.17733):

```
Cortês, G., Lourenço, N., Machado, P. (2024). Towards Physical Plausibility in Neuroevolution Systems. In: Smith, S., Correia, J., Cintrano, C. (eds) Applications of Evolutionary Computation. EvoApplications 2024. Lecture Notes in Computer Science, vol 14635. Springer, Cham. https://doi.org/10.1007/978-3-031-56855-8_5
```

## Citations

If you benefit from this project or make use of its code, concepts, or materials, please consider citing the following references.

```
Cortês, G., Lourenço, N., & Machado, P. (2024). ENERGIZE (v1.0). Zenodo. https://doi.org/10.5281/zenodo.11220573
```
