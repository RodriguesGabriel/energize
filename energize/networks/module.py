import random
import logging
from typing import Dict, List, TYPE_CHECKING

from energize.networks.module_config import ModuleConfig

if TYPE_CHECKING:
    from energize.evolution.grammar import Genotype, Grammar

logger = logging.getLogger(__name__)


class Module:
    def __init__(self, module_name: str, module_configuration: ModuleConfig) -> None:
        self.module_name: str = module_name
        self.module_configuration: ModuleConfig = module_configuration
        self.layers: List[Genotype] = []
        self.connections: Dict[int, List[int]] = {}

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

    def add_layer(self, individual_idx: int,  module_idx: int, grammar: 'Grammar', reuse_layer_prob: float):
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

    def remove_layer(self, individual_idx: int, module_idx: int):
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

    def layer_dsge(self, individual_idx: int, module_idx: int, grammar: 'Grammar', layer_idx: int):
        from energize.evolution.operators import mutation_dsge
        mutation_dsge(self.layers[layer_idx], grammar)
        logger.info("Individual %d is going to have a DSGE mutation on Module %d: %s; position %d",
                    individual_idx, module_idx, self.module_name, layer_idx)

    def layer_add_connection(self, individual_idx: int, module_idx: int, layer_idx: int):
        connection_possibilities = list(range(max(0, layer_idx-self.module_configuration.levels_back),
                                              layer_idx-1))
        connection_possibilities = list(
            set(connection_possibilities) - set(self.connections[layer_idx]))
        if len(connection_possibilities) > 0:
            new_input: int = random.choice(connection_possibilities)
            self.connections[layer_idx].append(new_input)
        logger.info("Individual %d is going to have a new connection Module %d: %s; layer %d",
                    individual_idx, module_idx, self.module_name, layer_idx)

    def layer_remove_connection(self, individual_idx: int, module_idx: int, layer_idx: int):
        connection_possibilities = list(
            set(self.connections[layer_idx]) - set([layer_idx-1]))
        if len(connection_possibilities) > 0:
            r_connection = random.choice(connection_possibilities)
            self.connections[layer_idx].remove(r_connection)
        logger.info("Individual %d is going to have a connection removed from Module %d: %s; layer %d",
                    individual_idx, module_idx, self.module_name, layer_idx)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Module):
            return self.__dict__ == other.__dict__
        return False
