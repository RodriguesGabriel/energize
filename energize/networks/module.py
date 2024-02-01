import logging
import random
import time
from copy import deepcopy
from sys import float_info
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Size, nn

from energize.misc.constants import DATASETS_INFO
from energize.misc.enums import Device
from energize.misc.power import PowerConfig
from energize.misc.utils import InvalidNetwork
from energize.networks.module_config import ModuleConfig
from energize.networks.torch.evaluators import (BaseEvaluator, LegacyEvaluator,
                                                parse_phenotype)
from energize.networks.torch.model_builder import ModelBuilder

if TYPE_CHECKING:
    from energize.evolution.grammar import Genotype, Grammar

logger = logging.getLogger(__name__)


class Module:
    history: List['Module'] = []
    power_config: PowerConfig = None

    def __init__(self, module_name: str, module_configuration: ModuleConfig) -> None:
        self.module_name: str = module_name
        self.module_configuration: ModuleConfig = module_configuration
        self.layers: List[Genotype] = []
        self.connections: Dict[int, List[int]] = {}
        self.power: Optional[float] = None

    def initialise(self, grammar: 'Grammar', reuse: float) -> None:
        num_expansions = random.choice(
            self.module_configuration.initial_network_structure)

        # Initialise layers
        for idx in range(num_expansions):
            if idx > 0 and random.random() <= reuse:
                r_idx = random.randint(0, idx-1)
                self.layers.append(self.layers[r_idx])
            else:
                self.layers.append(grammar.initialise(self.module_name))
        # Initialise connections: feed-forward and allowing skip-connections
        self.connections = {}

        for layer_idx in range(num_expansions):
            if layer_idx == 0:
                # the -1 layer is the input
                self.connections[layer_idx] = [-1,]
            else:
                connection_possibilities = list(range(max(0, layer_idx-self.module_configuration.levels_back),
                                                      layer_idx-1))
                if len(connection_possibilities) < self.module_configuration.levels_back-1:
                    connection_possibilities.append(-1)

                sample_size = random.randint(0, len(connection_possibilities))

                self.connections[layer_idx] = [layer_idx-1]
                if sample_size > 0:
                    self.connections[layer_idx] += random.sample(
                        connection_possibilities, sample_size)

        self.measure_power(grammar)

    def decode(self, grammar: 'Grammar', layer_counter: int) -> Tuple[int, str]:
        phenotype: str = ''
        offset: int = layer_counter
        for layer_idx, layer_genotype in enumerate(self.layers):
            layer_counter += 1
            phenotype_layer: str = f" {grammar.decode(self.module_name, layer_genotype)}"
            current_connections = deepcopy(self.connections[layer_idx])
            # ADRIANO HACK
            if "relu_agg" in phenotype_layer and -1 not in self.connections[layer_idx]:
                current_connections = [-1] + current_connections
            # END
            phenotype += (
                f"{phenotype_layer}"
                f" input:{','.join(map(str, np.array(current_connections) + offset))}"
            )
        return layer_counter, phenotype

    def measure_power(self, grammar: 'Grammar') -> None:
        if self.power_config is None or not self.power_config.measure_modules_power:
            return

        phenotype = self.decode(grammar, 0)[1][1:]
        parsed_network, _ = parse_phenotype(phenotype)
        device = BaseEvaluator.decide_device(Device.GPU)

        try:
            input_size: Tuple[int, int, int] = (1, 32, 32)
            model_builder: ModelBuilder = ModelBuilder(
                parsed_network, device, Size(list(input_size)))
            torch_model = model_builder.assemble_network(LegacyEvaluator)

            BaseEvaluator.adapt_model_to_device(torch_model, device)

            trainable_params_count: int = sum(
                p.numel() for p in torch_model.parameters() if p.requires_grad)
            if trainable_params_count == 0:
                raise InvalidNetwork(
                    "Network does not contain any trainable parameters.")

            torch_model.eval()
            with torch.no_grad():
                random_input = torch.rand(
                    10**4, *input_size).to(device.value, non_blocking=True)
                self.power_config.meter.start(tag="module")
                for _ in range(10**2):
                    torch_model(random_input)
                self.power_config.meter.stop()

            trace = self.power_config.meter.get_trace()
            self.power = sum(trace[0].energy.values()) / \
                1000 / trace[0].duration
            if self.power == 0:
                logger.warning("Module power is zero. Not saving it.")
            else:
                self.history.append(deepcopy(self))
        except InvalidNetwork as e:
            logger.warning("Invalid network: %s", e)
        except RuntimeError as e:
            logger.warning("Runtime error: %s", e)

    def add_layer(self, grammar: 'Grammar', individual_idx: int,  module_idx: int, reuse_layer_prob: float):
        if len(self.layers) >= self.module_configuration.max_expansions:
            return

        if random.random() <= reuse_layer_prob and len(self.layers) > 0:
            new_layer = random.choice(self.layers)
        else:
            new_layer = grammar.initialise(self.module_name)

        insert_pos: int = random.randint(0, len(self.layers))
        # fix connections
        for _key_ in sorted(self.connections, reverse=True):
            if _key_ >= insert_pos:
                for value_idx, value in enumerate(self.connections[_key_]):
                    if value >= insert_pos-1:
                        self.connections[_key_][value_idx] += 1

                self.connections[_key_ + 1] = self.connections.pop(_key_)

        self.layers.insert(insert_pos, new_layer)

        # make connections of the new layer
        if insert_pos == 0:
            self.connections[insert_pos] = [-1]
        else:
            connection_possibilities = list(range(max(0, insert_pos-self.module_configuration.levels_back),
                                                  insert_pos-1))
            if len(connection_possibilities) < self.module_configuration.levels_back-1:
                connection_possibilities.append(-1)

            sample_size = random.randint(
                0, len(connection_possibilities))

            self.connections[insert_pos] = [insert_pos-1]
            if sample_size > 0:
                self.connections[insert_pos] += random.sample(
                    connection_possibilities, sample_size)

        logger.info("Individual %d is going to have an extra layer at Module %d: %s; position %d",
                    individual_idx, module_idx, self.module_name, insert_pos)
        self.measure_power(grammar)

    def remove_layer(self, grammar: 'Grammar', individual_idx: int, module_idx: int):
        if len(self.layers) <= self.module_configuration.min_expansions:
            return
        remove_idx = random.randint(0, len(self.layers)-1)
        del self.layers[remove_idx]

        # fix connections
        if remove_idx == max(self.connections.keys()):
            self.connections.pop(remove_idx)
        else:
            for _key_ in sorted(self.connections.keys()):
                if _key_ > remove_idx:
                    if _key_ > remove_idx+1 and remove_idx in self.connections[_key_]:
                        self.connections[_key_].remove(remove_idx)

                    for value_idx, value in enumerate(self.connections[_key_]):
                        if value >= remove_idx:
                            self.connections[_key_][value_idx] -= 1
                    self.connections[_key_ -
                                     1] = list(set(self.connections.pop(_key_)))
            if remove_idx == 0:
                self.connections[0] = [-1]
        logger.info("Individual %d is going to have a layer removed from Module %d: %s; position %d",
                    individual_idx, module_idx, self.module_name, remove_idx)
        self.measure_power(grammar)

    def layer_dsge(self, grammar: 'Grammar', individual_idx: int, module_idx: int, layer_idx: int):
        from energize.evolution.operators import mutation_dsge
        mutation_dsge(self.layers[layer_idx], grammar)
        logger.info("Individual %d is going to have a DSGE mutation on Module %d: %s; position %d",
                    individual_idx, module_idx, self.module_name, layer_idx)
        self.measure_power(grammar)

    def layer_add_connection(self, grammar: 'Grammar', individual_idx: int, module_idx: int, layer_idx: int):
        connection_possibilities = list(range(max(0, layer_idx-self.module_configuration.levels_back),
                                              layer_idx-1))
        connection_possibilities = list(
            set(connection_possibilities) - set(self.connections[layer_idx]))
        if len(connection_possibilities) == 0:
            return
        new_input: int = random.choice(connection_possibilities)
        self.connections[layer_idx].append(new_input)
        logger.info("Individual %d is going to have a new connection Module %d: %s; layer %d",
                    individual_idx, module_idx, self.module_name, layer_idx)
        self.measure_power(grammar)

    def layer_remove_connection(self, grammar: 'Grammar', individual_idx: int, module_idx: int, layer_idx: int):
        connection_possibilities = list(
            set(self.connections[layer_idx]) - set([layer_idx-1]))
        if len(connection_possibilities) == 0:
            return
        r_connection = random.choice(connection_possibilities)
        self.connections[layer_idx].remove(r_connection)
        logger.info("Individual %d is going to have a connection removed from Module %d: %s; layer %d",
                    individual_idx, module_idx, self.module_name, layer_idx)
        self.measure_power(grammar)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Module):
            return self.__dict__ == other.__dict__
        return False
