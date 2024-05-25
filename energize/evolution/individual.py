from __future__ import annotations

import logging
import re
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from energize.misc.enums import Entity, Mutation, MutationType
from energize.misc.evaluation_metrics import EvaluationMetrics
from energize.misc.fitness_metrics import Fitness
from energize.networks import Module

if TYPE_CHECKING:
    from energize.evolution.grammar import Genotype, Grammar
    from energize.networks.module import ModuleConfig
    from energize.networks.torch.evaluators import BaseEvaluator


__all__ = ['Individual']


logger = logging.getLogger(__name__)


class Individual:
    """
        Candidate solution.


        Attributes
        ----------
        network_structure : list
            ordered list of tuples formated as follows
            [(non-terminal, min_expansions, max_expansions), ...]

        output_rule : str
            output non-terminal symbol

        macro_rules : list
            list of non-terminals (str) with the marco rules (e.g., learning)

        modules : list
            list of Modules (genotype) of the layers

        output : dict
            output rule genotype

        macro : list
            list of Modules (genotype) for the macro rules

        phenotype : str
            phenotype of the candidate solution

        fitness : float
            fitness value of the candidate solution

        metrics : dict
            training metrics

        num_epochs : int
            number of performed epochs during training

        trainable_parameters : int
            number of trainable parameters of the network

        time : float
            network training time

        current_time : float
            performed network training time

        train_time : float
            maximum training time

        id : int
            individual unique identifier


        Methods
        -------
            initialise(grammar, levels_back, reuse)
                Randomly creates a candidate solution

            decode(grammar)
                Maps the genotype to the phenotype

            evaluate(grammar, cnn_eval, weights_save_path, parent_weights_path='')
                Performs the evaluation of a candidate solution
    """

    def __init__(self, network_architecture_config: Dict[str, Any], ind_id: int, track_mutations: bool, seed: int) -> None:

        self.seed: int = seed
        self.modules_configurations: Dict[str,
                                          ModuleConfig] = network_architecture_config['modules']
        self.output_rule: str = network_architecture_config['output']
        self.macro_rules: List[str] = network_architecture_config['macro_structure']
        self.modules: List[Module] = []
        self.modules_phenotypes = []
        self.mutation_tracker: Optional[List[Mutation]] = [
        ] if track_mutations else None
        self.output: Optional[Genotype] = None
        self.macro: List[Genotype] = []
        self.phenotype: Optional[str] = None
        self.fitness: Optional[Fitness] = None
        self.metrics: Optional[EvaluationMetrics] = None
        self.num_epochs: int = 0
        self.current_time: float = 0.0
        self.total_allocated_train_time: float = 0.0
        self.total_training_time_spent: float = 0.0
        self.id: int = ind_id

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Individual):
            return self.__dict__ == other.__dict__
        return False

    def initialise(self, grammar: Grammar, reuse: float) -> "Individual":
        """
            Randomly creates a candidate solution

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            levels_back : dict
                number of previous layers a given layer can receive as input

            reuse : float
                likelihood of reusing an existing layer

            Returns
            -------
            candidate_solution : Individual
                randomly created candidate solution
        """
        for module_name, module_config in self.modules_configurations.items():
            new_module: Module = Module(module_name, module_config)
            new_module.initialise(grammar, reuse)
            self.modules.append(new_module)

        # Initialise output
        self.output = grammar.initialise(self.output_rule)

        dynamic_bounds = {
            'partition_point': (0, self.get_num_layers() - 1)
        }
        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule, dynamic_bounds))

        return self

    def load(self, grammar: Grammar, phenotype: str, config_macro_structure: list, weights_path: Optional[str] = None) -> "Individual":
        assert not self.modules
        assert self.id == 0
        SPLIT_LAYERS_PATTERN = r'\s(?=layer:)'

        params = re.split(fr"(?={'|'.join(config_macro_structure)})", phenotype)
        params = tuple(i.strip() for i in params)
        if '|' in params[0]:
            modules = params[0].split(' | ')
            # get the output layer from the last module
            last_module_layers = re.split(SPLIT_LAYERS_PATTERN, modules[-1])
            output_layer = last_module_layers[-1]
            modules[-1] = ' '.join(last_module_layers[:-1])
        else:
            modules = re.split(SPLIT_LAYERS_PATTERN, params[0])
            output_layer = modules[-1]
            modules = modules[:-1]
        # get the macro parameters
        macro = params[1:]

        for module in modules:
            new_module: Module = Module.load(grammar, module, self.modules_configurations)
            self.modules.append(new_module)

        self.output = grammar.encode(output_layer, self.output_rule)
        grammar.ensure_genotype_integrity(output_layer, self.output, self.output_rule)

        # TODO dynamic bounds?
        for m, m_rule in zip(macro, config_macro_structure):
            macro_geno = grammar.encode(m, m_rule)
            grammar.ensure_genotype_integrity(m, macro_geno, m_rule)
            self.macro.append(macro_geno)

        return self

    def get_num_layers(self) -> int:
        return sum(len(m.layers) for m in self.modules)

    def requires_dsge_mutation_macro(self, rule_idx: int, macro_rule: str) -> tuple[bool, str]:
        if macro_rule == "model_partition":
            partition_point = tuple(self.macro[rule_idx].expansions.values())[
                0][0][1].attribute.values
            if partition_point[0] >= self.get_num_layers():
                return True, "Mutation enforced because the partition point is greater than the number of layers"
        return False, None

    def _decode(self, grammar: Grammar) -> str:
        """
            Maps the genotype to the phenotype

            Parameters
            ----------
            grammar : Grammar
                grammar instaces that stores the expansion rules

            Returns
            -------
            phenotype : str
                phenotype of the individual to be used in the mapping to the keras model.
        """

        phenotype: str = ''
        layer_counter: int = 0
        for module in self.modules:
            layer_counter, module_phenotype = module.decode(
                grammar, layer_counter)
            phenotype += module_phenotype

        final_input_layer_id: int
        assert self.output is not None
        final_phenotype_layer: str = grammar.decode(
            self.output_rule, self.output)

        final_input_layer_id = layer_counter - 1

        phenotype += " " + final_phenotype_layer + " input:" + \
            str(final_input_layer_id)

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += " " + grammar.decode(macro_rule, self.macro[rule_idx])

        self.phenotype = phenotype.rstrip().lstrip()
        return self.phenotype

    def reuse_module(self, grammar: Grammar, generation: int):
        powers = np.array([module.power for module in Module.history])
        probs = 1 / powers
        probs /= probs.sum()
        chosen_module = np.random.choice(Module.history, p=probs)
        insert_idx: int = random.randint(0, len(self.modules))
        self.modules.insert(insert_idx, deepcopy(chosen_module))
        logger.info("Individual %d is going to have a module [%s] reused and inserted into position %d",
                    self.id, chosen_module.module_name, insert_idx)
        self.track_mutation(MutationType.REUSE_MODULE, generation, {
            "module": chosen_module.decode(grammar, 0)[1],
            "insert_idx": insert_idx
        })

    def remove_module(self, grammar: Grammar, generation: int) -> None:
        if len(self.modules) == 1:
            return

        remove_idx: int = random.randint(0, len(self.modules) - 1)
        logger.info("Individual %d is going to have a module [%s] removed from position %d",
                    self.id,  self.modules[remove_idx].module_name, remove_idx)
        self.track_mutation(MutationType.REMOVE_MODULE, generation, {
            "module": self.modules[remove_idx].decode(grammar, 0)[1],
            "remove_idx": remove_idx
        })
        self.modules.pop(remove_idx)

    def track_mutation(self, mutation_type: MutationType, gen: int, data: dict[str, Any]) -> None:
        if self.mutation_tracker is None:
            return
        self.mutation_tracker.append(Mutation(mutation_type, gen, data))

    def evaluate(self,
                 grammar: Grammar,
                 cnn_eval: BaseEvaluator,
                 generation: int,
                 model_saving_dir: str,
                 parent_dir: Optional[str] = None) -> Fitness:  # pragma: no cover

        phenotype: str
        phenotype = self._decode(grammar)

        # store modules phenotypes
        self.modules_phenotypes = [module.decode(
            grammar, 0)[1].lstrip() for module in self.modules]

        reuse_parent_weights: bool
        reuse_parent_weights = True
        if self.current_time == 0:
            reuse_parent_weights = False

        allocated_train_time: float = self.total_allocated_train_time - self.current_time
        logger.info("-----> Starting evaluation for individual %d for %d secs", self.id, allocated_train_time)

        first_individual_overall = generation == 0 and self.id == 0

        evaluation_metrics: EvaluationMetrics = cnn_eval.evaluate(phenotype,
                                                                  first_individual_overall,
                                                                  model_saving_dir,
                                                                  parent_dir,
                                                                  reuse_parent_weights,
                                                                  allocated_train_time,
                                                                  self.num_epochs)
        if self.metrics is None:
            self.metrics = evaluation_metrics
        else:
            self.metrics += evaluation_metrics
        self.fitness = self.metrics.fitness
        self.num_epochs += self.metrics.n_epochs
        self.current_time += allocated_train_time
        self.total_training_time_spent += self.metrics.training_time_spent

        assert self.fitness is not None
        return self.fitness
