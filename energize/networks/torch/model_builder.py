from __future__ import annotations

import logging
import warnings
from collections import Counter
from math import ceil
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Size, Tensor, nn, optim

from energize.misc.constants import SEPARATOR_CHAR
from energize.misc.enums import (ActivationType, Device, LayerType,
                                 OptimiserType)
from energize.misc.phenotype_parser import Layer
from energize.misc.utils import InputLayerId, InvalidNetwork, LayerId
from energize.networks import Dimensions
from energize.networks.torch import (LARS, BaseEvaluator, EvolvedNetwork,
                                     LearningParams, LegacyEvaluator,
                                     LegacyNetwork)

warnings.filterwarnings("ignore")


if TYPE_CHECKING:
    from energize.misc.phenotype_parser import Optimiser, ParsedNetwork


logger = logging.getLogger(__name__)


class ModelBuilder():
    def __init__(self,
                 parsed_network: ParsedNetwork,
                 device: Device,
                 input_shape: Size) -> None:
        self.parsed_network: ParsedNetwork = parsed_network
        self.layer_shapes: Dict[InputLayerId, Dimensions] = {
            InputLayerId(-1): Dimensions(channels=input_shape[0],
                                         height=input_shape[1],
                                         width=input_shape[2])
        }
        self.layer_type_counts: Counter = Counter([])
        self.device = device
        self.additional_output_idx: List[LayerId] = []

    @classmethod
    def assemble_optimiser(cls, model_parameters: Iterable[Tensor], optimiser: Optimiser) -> LearningParams:
        early_stop: Optional[int] = optimiser.optimiser_parameters.pop(
            "early_stop", None)
        batch_size: int = optimiser.optimiser_parameters.pop("batch_size")
        epochs: int = optimiser.optimiser_parameters.pop("epochs")
        torch_optimiser: optim.Optimizer
        if optimiser.optimiser_type == OptimiserType.RMSPROP:
            torch_optimiser = optim.RMSprop(
                params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.GRADIENT_DESCENT:
            torch_optimiser = optim.SGD(
                params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.ADAM:
            optimiser.optimiser_parameters["betas"] = (
                optimiser.optimiser_parameters.pop("beta1"),
                optimiser.optimiser_parameters.pop("beta2")
            )
            torch_optimiser = optim.Adam(
                params=model_parameters, **optimiser.optimiser_parameters)
        elif optimiser.optimiser_type == OptimiserType.LARS:
            param_weights = []
            param_biases = []
            for param in model_parameters:
                if param.ndim == 1:
                    param_biases.append(param)
                else:
                    param_weights.append(param)
            parameters = [{'params': param_weights}, {'params': param_biases}]
            torch_optimiser = LARS(parameters,
                                   batch_size=batch_size,
                                   eta=0.001,
                                   weight_decay_filter=True,
                                   lars_adaptation_filter=True,
                                   **optimiser.optimiser_parameters)
        else:
            raise ValueError(
                f"Invalid optimiser name found: {optimiser.optimiser_type}")
        return LearningParams(
            early_stop=early_stop,
            batch_size=batch_size,
            epochs=epochs,
            torch_optimiser=torch_optimiser
        )

    def assemble_network(self,
                         evaluation_type: type[BaseEvaluator]) -> EvolvedNetwork:
        layer_to_add: nn.Module
        torch_layers: List[Tuple[str, nn.Module]] = []
        collected_extra_torch_layers: List[Tuple[str, nn.Module]] = []
        self.layer_type_counts.update(list(LayerType))

        # add additional output layer
        if self.parsed_network.model_partition_points is not None:
            assert self.parsed_network.layers[-1].layer_type == LayerType.FC
            layer_id = LayerId(len(self.parsed_network.layers))
            layer = Layer(layer_id,
                          layer_type=LayerType('fc'),
                          layer_parameters={
                              "act": "softmax",
                              "out_features": self.parsed_network.layers[-1].layer_parameters['out_features'],
                              "bias": "True"
                          })
            self.parsed_network.layers_connections[layer_id] = [
                self.parsed_network.model_partition_points]
            self.parsed_network.layers.append(layer)
            self.additional_output_idx.append(layer_id)

        connections_to_use: Dict[LayerId, List[InputLayerId]
                                 ] = self.parsed_network.layers_connections

        try:
            for i, l in enumerate(self.parsed_network.layers):
                layer_name: str = f"{l.layer_type.value}{SEPARATOR_CHAR}{self.layer_type_counts[l.layer_type]}"
                self.layer_type_counts.update([l.layer_type])

                inputs_shapes: Dict[InputLayerId, Dimensions] = \
                    {input_id: self.layer_shapes[input_id]
                     for input_id in connections_to_use[LayerId(i)]}
                minimum_extra_id: int = len(
                    self.parsed_network.layers) + len(collected_extra_torch_layers)
                extra_layers: Dict[InputLayerId, Layer] = \
                    self._get_layers_to_fix_shape_mismatches_resnet_approach(
                        inputs_shapes, minimum_extra_id)
                for input_id, extra_layer in extra_layers.items():
                    extra_layer_name: str = f"{extra_layer.layer_type.value}{SEPARATOR_CHAR}" + \
                        f"{self.layer_type_counts[extra_layer.layer_type]}"
                    self.layer_type_counts.update([extra_layer.layer_type])

                    # Add the new connection in between
                    connections_to_use[LayerId(extra_layer.layer_id)] = [
                        input_id]
                    # Fix the old ones
                    connections_to_use[LayerId(i)].remove(input_id)
                    connections_to_use[LayerId(i)].append(
                        InputLayerId(extra_layer.layer_id))

                    layer_to_add = self._create_torch_layer(
                        extra_layer, extra_layer_name, False)
                    collected_extra_torch_layers.append(
                        (extra_layer_name, layer_to_add))
                layer_to_add = self._create_torch_layer(
                    l, layer_name, l.layer_id in self.additional_output_idx)
                torch_layers.append((layer_name, layer_to_add))

            if evaluation_type is LegacyEvaluator:
                output_layer_idx = self.parsed_network.get_output_layer_idx()
                # TODO check with assert if len(^^) should be 1 or not
                return LegacyNetwork(torch_layers + collected_extra_torch_layers,
                                     connections_to_use,
                                     output_layer_idx)
            raise ValueError(f"Unexpected network type: {evaluation_type}")
        except InvalidNetwork as e:
            raise e
        except RuntimeError as e:
            print(e)
            raise InvalidNetwork(str(e)) from e

    def _get_layers_to_fix_shape_mismatches_resnet_approach(self,
                                                            inputs_shapes: Dict[InputLayerId, Dimensions],
                                                            minimum_extra_id: int) -> Dict[InputLayerId, Layer]:
        # assumes that of the inputs will be the beginning of the module, just like a resnet block
        new_layers_dict: Dict[InputLayerId, Layer] = {}
        minimum_input_id: InputLayerId = min(list(inputs_shapes.keys()))
        target_shape: Dimensions = inputs_shapes.pop(
            minimum_input_id)  # side effect: removes element from dict
        layer_id: int = minimum_extra_id
        for input_id, input_shape in inputs_shapes.items():
            if input_shape != target_shape:
                logging.warning("Shape mismatch found")
                expected_padding_h = max(
                    ceil((target_shape.height-input_shape.height) / 2), 0)
                expected_padding_w = max(
                    ceil((target_shape.width-input_shape.width) / 2), 0)
                expected_kernel_size_h = max(
                    ceil(input_shape.height + (expected_padding_h * 2) -
                         target_shape.height + 1),
                    1
                )
                expected_kernel_size_w = max(
                    ceil(input_shape.height + (expected_padding_w * 2) -
                         target_shape.width + 1),
                    1
                )
                new_layers_dict[input_id] = \
                    Layer(layer_id=LayerId(layer_id),
                          layer_type=LayerType.CONV,
                          layer_parameters={
                        'out_channels': str(target_shape.channels),
                        'kernel_size_fix': str((expected_kernel_size_h, expected_kernel_size_w)),
                        'padding_fix': str((expected_padding_h, expected_padding_w)),
                        'act': 'linear',
                        'stride': '1'})
                layer_id += 1
        return new_layers_dict

    def _get_layers_to_fix_shape_mismatches(self,
                                            inputs_shapes: Dict[InputLayerId, Dimensions],
                                            minimum_extra_id: int) -> Dict[InputLayerId, Layer]:
        new_layers_dict: Dict[InputLayerId, Layer] = {}

        # max_channels: int = min(list(map(lambda x: x.channels, inputs_shapes.values())))
        min_height: int = min(
            list(map(lambda x: x.height, inputs_shapes.values())))
        min_width: int = min(
            list(map(lambda x: x.width, inputs_shapes.values())))
        layer_id: int = minimum_extra_id
        for input_id, input_shape in inputs_shapes.items():
            if input_shape.height > min_height or input_shape.width > min_width:
                logging.warning("Shape mismatch found")
                new_layers_dict[input_id] = \
                    Layer(layer_id=LayerId(layer_id),
                          layer_type=LayerType.POOL_MAX,
                          layer_parameters={'kernel_size_fix': str((input_shape.height-min_height + 1,
                                                                    input_shape.width-min_width + 1)),
                                            'padding': 'valid',
                                            'stride': '1'})
                layer_id += 1
        return new_layers_dict

    def _create_activation_layer(self, activation: ActivationType) -> nn.Module:
        if activation == ActivationType.RELU:
            return nn.ReLU()
        if activation == ActivationType.SIGMOID:
            return nn.Sigmoid()
        if activation == ActivationType.SOFTMAX:
            return nn.Softmax()
        raise ValueError(
            f"Unexpected activation function found: {activation}")

    def _create_torch_layer(self, layer: Layer, layer_name: str, require_flatten: bool) -> nn.Module:
        inputs_shapes = {input_id: self.layer_shapes[input_id]
                         for input_id in self.parsed_network.layers_connections[layer.layer_id]}
        expected_input_dimensions: Optional[Dimensions]
        # if len(set(inputs_shapes.values())) == 1:
        first_input = list(inputs_shapes.values())[0]
        if len(inputs_shapes) > 1:
            # total_channels: int = sum([x.channels for x in list(inputs_shapes.values())])
            # expected_input_dimensions = Dimensions(total_channels, first_input.height, first_input.width)
            # ADRIANO this is a test for the new shape mismatch resolution strategy
            expected_input_dimensions = Dimensions(
                first_input.channels, first_input.height, first_input.width)
        else:
            # in this case all inputs will have the same dimensions, just take the first one...
            expected_input_dimensions = first_input
        self.layer_shapes[InputLayerId(layer.layer_id)] = \
            Dimensions.from_layer(layer, expected_input_dimensions)
        if layer.layer_type == LayerType.CONV:
            return self._build_convolutional_layer(layer, expected_input_dimensions)
        if layer.layer_type == LayerType.BATCH_NORM:
            return self._build_batch_norm_layer(layer, expected_input_dimensions)
        if layer.layer_type == LayerType.POOL_AVG:
            return self._build_avg_pooling_layer(layer, expected_input_dimensions)
        if layer.layer_type == LayerType.POOL_MAX:
            return self._build_max_pooling_layer(layer, expected_input_dimensions)
        if layer.layer_type == LayerType.FC:
            return self._build_dense_layer(layer, layer_name, expected_input_dimensions, require_flatten)
        if layer.layer_type == LayerType.DROPOUT:
            return self._build_dropout_layer(layer)
        if layer.layer_type == LayerType.IDENTITY:
            return nn.Identity()
        if layer.layer_type == LayerType.RELU_AGG:
            return nn.ReLU()
        raise ValueError(
            f"Unexpected layer type found: {layer.layer_type}")

    def _build_convolutional_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        # pylint: disable=unused-variable
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.bias.data.fill_(0.01)

        layer_to_add: nn.Module
        activation: ActivationType = ActivationType(
            layer.layer_parameters.pop("act"))

        assert layer.layer_parameters['padding'] in ['valid', 'same'] or \
            isinstance(layer.layer_parameters['padding'], tuple)

        # TODO: put bias = False if all next layers connected to this layer are batch norm

        # If padding = same torch does not support strided convolutions.
        # Check https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        if layer.layer_parameters['padding'] == "same":
            layer.layer_parameters['stride'] = 1
        if activation == ActivationType.LINEAR:
            layer_to_add = nn.Conv2d(**layer.layer_parameters,
                                     in_channels=input_dimensions.channels,
                                     device=self.device.value)
            # torch.nn.init.xavier_uniform(layer_to_add.weight)
        else:
            layer_to_add = nn.Sequential(nn.Conv2d(**layer.layer_parameters,
                                                   in_channels=input_dimensions.channels,
                                                   device=self.device.value),
                                         self._create_activation_layer(activation))
            # layer_to_add.apply(init_weights)
        return layer_to_add

    def _build_batch_norm_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        return nn.BatchNorm2d(**layer.layer_parameters,
                              num_features=input_dimensions.channels,
                              device=self.device.value)

    def _build_avg_pooling_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        torch_layers_to_add: List[nn.Module] = []
        padding_type: str = layer.layer_parameters.pop("padding")
        layer.layer_parameters['padding'] = 0
        torch_layers_to_add.append(nn.AvgPool2d(**layer.layer_parameters))
        if padding_type == "same":
            # https://github.com/pytorch/pytorch/issues/3298
            padding_to_apply: Tuple[int, int, int, int] = \
                input_dimensions.compute_adjusting_padding(layer.layer_parameters['kernel_size'],
                                                           layer.layer_parameters['stride'])
            torch_layers_to_add.append(nn.ZeroPad2d(padding_to_apply))
        return nn.Sequential(*torch_layers_to_add)

    def _build_max_pooling_layer(self, layer: Layer, input_dimensions: Dimensions) -> nn.Module:
        torch_layers_to_add: List[nn.Module] = []
        padding_type: str = layer.layer_parameters.pop("padding")
        layer.layer_parameters['padding'] = 0
        torch_layers_to_add.append(nn.MaxPool2d(**layer.layer_parameters))
        if padding_type == "same":
            # https://github.com/pytorch/pytorch/issues/3298
            padding_to_apply: Tuple[int, int, int, int] = \
                input_dimensions.compute_adjusting_padding(layer.layer_parameters['kernel_size'],
                                                           layer.layer_parameters['stride'])
            torch_layers_to_add.append(nn.ZeroPad2d(padding_to_apply))
        return nn.Sequential(*torch_layers_to_add)

    def _build_dropout_layer(self, layer: Layer) -> nn.Module:
        layer.layer_parameters['p'] = layer.layer_parameters.pop("rate")
        return nn.Dropout(**layer.layer_parameters)

    def _build_dense_layer(self, layer: Layer, layer_name: str, input_dimensions: Dimensions, require_flatten: bool) -> nn.Sequential:
        activation = ActivationType(layer.layer_parameters.pop("act"))
        torch_layers_to_add: List[nn.Module] = []
        if require_flatten or layer_name.endswith(f"{LayerType.FC.value}{SEPARATOR_CHAR}1"):
            torch_layers_to_add.append(nn.Flatten())
            torch_layers_to_add.append(nn.Linear(**layer.layer_parameters,
                                                 in_features=input_dimensions.flatten(),
                                                 device=self.device.value))
            # torch.nn.init.xavier_uniform(torch_layers_to_add[1].weight)
            # torch_layers_to_add[1].bias.data.fill_(0.01)
        else:
            torch_layers_to_add.append(nn.Linear(**layer.layer_parameters,
                                                 in_features=input_dimensions.channels,
                                                 device=self.device.value))
            # torch.nn.init.xavier_uniform(torch_layers_to_add[0].weight)
            # torch_layers_to_add[0].bias.data.fill_(0.01)
        if activation != ActivationType.LINEAR:
            torch_layers_to_add.append(
                self._create_activation_layer(activation))

        return nn.Sequential(*torch_layers_to_add)
