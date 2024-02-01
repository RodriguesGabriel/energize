import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import torch
from numpy import add
from torch import Tensor, nn

from energize.misc.constants import SEPARATOR_CHAR
from energize.misc.enums import Device, LayerType
from energize.misc.utils import InputLayerId, LayerId
from energize.networks import Dimensions

logger = logging.getLogger(__name__)


class EvolvedNetwork(nn.Module):
    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_idx: List[LayerId]) -> None:

        super().__init__()
        self.cache: Dict[Tuple[InputLayerId, LayerId], Tensor] = {}
        self.evolved_layers: List[Tuple[str, nn.Module]] = evolved_layers
        self.layers_connections: Dict[LayerId,
                                      List[InputLayerId]] = layers_connections
        self.output_layer_idx: List[LayerId] = output_layer_idx
        self.id_layername_map: Dict[LayerId, str] = {
            LayerId(i): l[0] for i, l in enumerate(evolved_layers)}

        for (layer_name, layer) in evolved_layers:
            setattr(self, layer_name, layer)

    def _clear_cache(self) -> None:
        self.cache.clear()
        torch.cuda.empty_cache()

    def _process_forward_pass(self,
                              x: Tensor,
                              layer_id: LayerId,
                              input_ids: List[InputLayerId]) -> Tensor:

        assert len(input_ids) > 0
        final_input_tensor: Tensor
        input_tensor: Tensor
        output_tensor: Tensor
        layer_name: str = self.id_layername_map[layer_id]
        layer_inputs = []
        for i in input_ids:
            if i == -1:
                input_tensor = x
            else:
                if (i, layer_id) in self.cache.keys():
                    input_tensor = self.cache[(i, layer_id)]
                else:
                    input_tensor = self._process_forward_pass(
                        x, LayerId(i), self.layers_connections[LayerId(i)])
                    self.cache[(i, layer_id)] = input_tensor
            layer_inputs.append(input_tensor)

        del input_tensor
        self._clear_cache()
        if len(layer_inputs) > 1:
            # we are using channels first representation, so channels is index 1
            # ADRIANO: another hack to cope with the relu in resnet scenario
            final_input_tensor = torch.stack(layer_inputs, dim=0).sum(dim=0)
            # old way: final_input_tensor = torch.cat(tuple(layer_inputs), dim=CHANNEL_INDEX)
        else:
            final_input_tensor = layer_inputs[0]
        del layer_inputs
        output_tensor = self.__dict__[
            '_modules'][layer_name](final_input_tensor)
        return output_tensor

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, x: Tensor) -> Optional[Tensor] | Optional[tuple[Tensor]]:
        input_layer_idx: List[InputLayerId]
        result: List[Tensor] = []
        for output_layer_id in self.output_layer_idx:
            input_layer_idx = self.layers_connections[output_layer_id]
            result.append(self._process_forward_pass(
                x, output_layer_id, input_layer_idx))
        return tuple(result) if len(result) > 1 else result[0]

    def remove_additional_outputs(self, additional_output_idx: List[LayerId]) -> 'EvolvedNetwork':
        model_partition: EvolvedNetwork = deepcopy(self)

        original_num_layers = len(model_partition.evolved_layers)
        layer_names_to_remove = []

        for additional_output_id in additional_output_idx:
            layer_name = model_partition.id_layername_map[additional_output_id]
            layer_names_to_remove.append(layer_name)
            model_partition.output_layer_idx.remove(additional_output_id)
            del model_partition.layers_connections[additional_output_id]
            del model_partition.id_layername_map[additional_output_id]
            delattr(model_partition, layer_name)

        model_partition.evolved_layers = [
            layer for layer in model_partition.evolved_layers
            if layer[0] not in layer_names_to_remove
        ]

        assert len(model_partition.evolved_layers) == original_num_layers - \
            len(additional_output_idx)
        assert len(model_partition.output_layer_idx) == 1

        return model_partition

    def get_connected_layers(self, layer_id: LayerId) -> set[LayerId]:
        connected_layers = set()
        for idx in self.layers_connections[layer_id]:
            connected_layers.add(idx)
            if idx != -1:
                connected_layers.update(self.get_connected_layers(idx))
        return connected_layers

    def prune_unnecessary_layers(self, additional_output_id: LayerId) -> 'EvolvedNetwork':
        model_partition: EvolvedNetwork = deepcopy(self)

        layers_to_keep = model_partition.get_connected_layers(
            additional_output_id)
        layers_to_keep.add(additional_output_id)

        layers_to_prune = set(model_partition.layers_connections.keys()) - \
            layers_to_keep

        layer_names_to_remove = []

        for idx in layers_to_prune:
            layer_name = model_partition.id_layername_map[idx]
            layer_names_to_remove.append(layer_name)
            del model_partition.layers_connections[idx]
            del model_partition.id_layername_map[idx]
            delattr(model_partition, layer_name)

        for connections in model_partition.layers_connections.values():
            for idx in connections:
                if idx in layers_to_prune:
                    connections.remove(idx)

        model_partition.evolved_layers = [
            layer for layer in model_partition.evolved_layers
            if layer[0] not in layer_names_to_remove
        ]

        model_partition.output_layer_idx = [additional_output_id]

        # check if the number of layers is right but don't consider the input layer (-1)
        assert len(model_partition.evolved_layers) == len(layers_to_keep) - 1
        assert len(model_partition.output_layer_idx) == 1

        return model_partition


class LegacyNetwork(EvolvedNetwork):
    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_idx: List[LayerId]) -> None:
        super().__init__(evolved_layers, layers_connections, output_layer_idx)

    def forward(self, x: Tensor) -> Optional[Tensor]:
        return super().forward(x)
