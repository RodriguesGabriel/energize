import re
from dataclasses import astuple, dataclass
from enum import Enum, unique
from typing import Any, Iterator, List, Optional

import torch


@unique
class AttributeType(Enum):
    INT = "int"
    INT_POWER2 = "int_power2"
    INT_POWER2_INV = "inv_power2"
    FLOAT = "float"


class ExtendedEnum(Enum):
    @classmethod
    def enum_values(cls) -> List[Any]:
        return list(map(lambda c: c.value, cls))  # type: ignore


@unique
class Entity(ExtendedEnum):
    LAYER = "layer"
    OPTIMISER = "learning"
    MODEL_PARTITION = "model_partition"


@unique
class Device(Enum):
    CPU = "cpu"
    GPU = "mps" if torch.backends.mps.is_available() else "cuda:0"


@unique
class LayerType(ExtendedEnum):
    CONV = "conv"
    BATCH_NORM = "batch_norm"
    BATCH_NORM_PROJ = "batch_norm_proj"
    POOL_AVG = "pool_avg"
    POOL_MAX = "pool_max"
    FC = "fc"
    DROPOUT = "dropout"
    IDENTITY = "identity"
    RELU_AGG = "relu_agg"


@unique
class OptimiserType(str, Enum):
    RMSPROP = "rmsprop"
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    LARS = "lars"


@unique
class ActivationType(Enum):
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


@unique
class TransformOperation(ExtendedEnum):
    COLOR_JITTER = "color_jitter"
    NORMALIZE = "normalize"
    PADDING = "padding"
    RANDOM_CROP = "random_crop"
    HORIZONTAL_FLIPPING = "random_horizontal_flip"
    RANDOM_RESIZED_CROP = "random_resized_crop"
    RANDOM_GRAYSCALE = "random_grayscale"
    GAUSSIAN_BLUR = "gaussian_blur"
    SOLARIZE = "random_solarize"
    CENTER_CROP = "center_crop"
    RESIZE = "resize"


@unique
class FitnessMetricName(ExtendedEnum):
    LOSS = "loss"
    ACCURACY = "accuracy"
    ACCURACY_N = r"accuracy_(\d+)"
    POWER = "power"
    POWER_N = r"power_(\d+)"
    ENERGY = "energy"
    ENERGY_N = r"energy_(\d+)"
    EVOZERO = "evozero"

    @staticmethod
    def new(value: str) -> tuple['FitnessMetricName', Optional[int]]:
        match: Optional[re.Match] = None
        if (match := re.match(FitnessMetricName.ACCURACY_N.value, value)):
            return FitnessMetricName.ACCURACY_N, int(match[1])
        if (match := re.match(FitnessMetricName.POWER_N.value, value)):
            return FitnessMetricName.POWER_N, int(match[1])
        if (match := re.match(FitnessMetricName.ENERGY_N.value, value)):
            return FitnessMetricName.ENERGY_N, int(match[1])
        return FitnessMetricName(value), None


class MutationType(ExtendedEnum):
    TRAIN_LONGER = "train_longer"
    REUSE_MODULE = "reuse_module"
    REMOVE_MODULE = "remove_module"
    ADD_LAYER = "add_layer"
    REUSE_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    DSGE_LAYER = "dsge_layer"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    DSGE_MACRO = "dsge_macro"

    def __str__(self) -> str:
        return self.value

@dataclass
class Mutation:
    mutation_type: MutationType
    gen: int
    data: dict

    def __str__(self) -> str:
        return f"Mutation type [{self.mutation_type}] on gen [{self.gen}] with data: {self.data}"

    def __iter__(self) -> Iterator[Any]:
        return iter(astuple(self))
