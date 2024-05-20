# Installation

The packages required to execute ENERGIZE can be installed through Conda or Pip.

### Conda

A conda environment can be created from an exported yaml file that contains all the required dependencies:

```
conda env create -f environment.yml
```

After the environment is created, activate it to be able to run your code:

```
conda activate energize
```

### pip

Alternatively, the `requirements.txt` file can be used. Note that CUDA Toolkit and other libraries that might be required to enable GPU acceleration are not installed this way.

```
pip install -r requirements.txt
```

**Note:** Installing ENERGIZE as a Python library is not yet supported
