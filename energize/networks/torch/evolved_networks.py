import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor

from energize.networks import Dimensions
from energize.misc.constants import SEPARATOR_CHAR
from energize.misc.enums import Device, LayerType
from energize.misc.utils import InputLayerId, LayerId


logger = logging.getLogger(__name__)


class EvolvedNetwork(nn.Module):
    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_id: LayerId) -> None:

        super().__init__()
        self.cache: Dict[Tuple[InputLayerId, LayerId], Tensor] = {}
        self.evolved_layers: List[Tuple[str, nn.Module]] = evolved_layers
        self.layers_connections: Dict[LayerId,
                                      List[InputLayerId]] = layers_connections
        self.output_layer_id: LayerId = output_layer_id
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
        # layer_outputs: List[Tensor] = []
        layer_name: str = self.id_layername_map[layer_id]
        layer_inputs = []
        for i in input_ids:
            if i == -1:
                input_tensor = x
                # print("---------- (end) processing layer: ", layer_id, input_tensor.shape)
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

    def forward(self, x: Tensor) -> Optional[Tensor]:
        input_layer_ids: List[InputLayerId]
        input_layer_ids = self.layers_connections[self.output_layer_id]
        return self._process_forward_pass(x, self.output_layer_id, input_layer_ids)


class LegacyNetwork(EvolvedNetwork):

    def __init__(self,
                 evolved_layers: List[Tuple[str, nn.Module]],
                 layers_connections: Dict[LayerId, List[InputLayerId]],
                 output_layer_id: LayerId) -> None:
        super().__init__(evolved_layers, layers_connections, output_layer_id)

    def forward(self, x: Tensor) -> Optional[Tensor]:
        return super().forward(x)
