from dataclasses import dataclass
from functools import reduce
from itertools import dropwhile, takewhile
from typing import Any, Dict, List, Optional, Set, Tuple, cast

from energize.misc.enums import Entity, LayerType, OptimiserType
from energize.misc.utils import InputLayerId, LayerId

import torch


class Layer:
    def __init__(self,
                 layer_id: LayerId,
                 layer_type: LayerType,
                 layer_parameters: Dict[str, str]) -> None:
        self.layer_id: LayerId = layer_id
        self.layer_type: LayerType = layer_type
        self.layer_parameters: Dict[str, Any] = dict(
            self._convert(k, v) for k, v in layer_parameters.items())

    def _convert(self, key: str, value: str) -> Tuple[str, Any]:
        if key == "bias":
            return key, value.title() == "True"
        if key in ["rate"]:
            return key, float(value)
        if key in ["out_channels", "out_features", "kernel_size", "stride"]:
            return key, int(value)
        if key in ["act", "padding"]:
            return key, value
        if key == "input":
            return key, list(map(int, value))
        if key == "kernel_size_fix":
            return "kernel_size", tuple(map(int, value[1:-1].split(',')))
        if key == "padding_fix":
            return "padding", tuple(map(int, value[1:-1].split(',')))
        raise ValueError(
            f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Layer):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        return f"Layer [{self.layer_type}] with id [{self.layer_id}] and params: {self.layer_parameters}"


@dataclass
class ParsedNetwork:
    layers: List[Layer]
    layers_connections: Dict[LayerId, List[InputLayerId]]
    model_partition_points: Optional[int] = None
    data_type: Optional[torch.dtype] = None

    # It gets the layer idx that corresponds to the final/output layer
    def get_output_layer_idx(self) -> List[LayerId]:
        keyset: Set[int] = set(self.layers_connections.keys())
        values_set: Set[int] = set(
            list(reduce(lambda a, b: cast(list, a) + cast(list, b),
                        self.layers_connections.values()))
        )
        result: Set[int] = keyset.difference(values_set)
        return list(sorted(map(LayerId, result)))


class Optimiser:
    def __init__(self,
                 optimiser_type: OptimiserType,
                 optimiser_parameters: Dict[str, str]) -> None:
        self.optimiser_type: OptimiserType = optimiser_type
        self.optimiser_parameters: Dict[str, Any] = {
            k: self._convert(k, v) for k, v in optimiser_parameters.items()
        }

    def _convert(self, key: str, value: str) -> Any:
        if key == "nesterov":
            return value.title() == "True"
        if key in ["lr", "lr_weights", "lr_biases", "alpha", "weight_decay", "momentum", "beta1", "beta2"]:
            return float(value)
        if key in ["early_stop", "batch_size", "epochs", "partition_point"]:
            return int(value)
        raise ValueError(
            f"No conversion found for param: [{key}], with value [{value}]")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Optimiser):
            return self.__dict__ == other.__dict__
        return False


DATA_TYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64
}


def parse_phenotype(phenotype: str) -> Tuple[ParsedNetwork, Optimiser]:
    # ignore modules separator
    phenotype = phenotype.replace("| ", "")
    phenotype_as_list: List[List[str]] = \
        list(map(lambda x: x.split(":"), phenotype.split(" ")))

    optimiser: Optimiser = None
    layers: List[Layer] = []
    layers_connections: Dict[LayerId, List[InputLayerId]] = {}
    layer_id: int = 0
    model_partition_point: Optional[int] = None
    data_type: Optional[torch.dtype] = None
    while phenotype_as_list:
        entity: Entity = Entity(phenotype_as_list[0][0])
        name: str = phenotype_as_list[0][1]
        entity_parameters: Dict[str, str] = {
            kv[0]: kv[1]
            for kv in takewhile(lambda kv: kv[0] not in Entity.enum_values(),
                                phenotype_as_list[1:])
        }
        phenotype_as_list = list(dropwhile(lambda kv: kv[0] not in Entity.enum_values(),
                                           phenotype_as_list[1:]))
        input_info: List[InputLayerId]
        if entity == Entity.LAYER:
            input_info = \
                list(map(lambda x: InputLayerId(int(x)),
                     entity_parameters.pop("input").split(",")))
            layers.append(Layer(LayerId(layer_id),
                                layer_type=LayerType(name),
                                layer_parameters=entity_parameters))
            layers_connections[LayerId(layer_id)] = input_info
            layer_id += 1
        elif entity == Entity.OPTIMISER:
            optimiser = Optimiser(optimiser_type=OptimiserType(name),
                                  optimiser_parameters=entity_parameters)
        elif entity == Entity.MODEL_PARTITION:
            model_partition_point = int(entity_parameters["partition_point"])
        elif entity == Entity.DATA_TYPE:
            data_type = DATA_TYPES[entity_parameters["data_type"]]
        else:
            raise ValueError(f"Unknown entity: {entity}")
    return ParsedNetwork(layers, layers_connections, model_partition_point, data_type), \
        optimiser
