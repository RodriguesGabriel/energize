from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from energize.networks import Module
from energize.misc.evaluation_metrics import EvaluationMetrics
from energize.misc.fitness_metrics import Fitness


if TYPE_CHECKING:
    from energize.evolution.grammar import Genotype, Grammar
    from energize.networks.torch.evaluators import BaseEvaluator
    from energize.networks.module import ModuleConfig

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

    def __init__(self, network_architecture_config: Dict[str, Any], ind_id: int, seed: int) -> None:

        self.seed: int = seed
        self.modules_configurations: Dict[str,
                                          ModuleConfig] = network_architecture_config['modules']
        self.output_rule: str = network_architecture_config['output']
        self.macro_rules: List[str] = network_architecture_config['macro_structure']
        self.modules: List[Module] = []
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

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self

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
            layer_counter, module_phenotype = module.decode(grammar, layer_counter)
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

    def evaluate(self,
                 grammar: Grammar,
                 cnn_eval: BaseEvaluator,
                 model_saving_dir: str,
                 parent_dir: Optional[str] = None) -> Fitness:  # pragma: no cover

        phenotype: str
        phenotype = self._decode(grammar)

        reuse_parent_weights: bool
        reuse_parent_weights = True
        if self.current_time == 0:
            reuse_parent_weights = False

        allocated_train_time: float = self.total_allocated_train_time - self.current_time
        logger.info(
            f"-----> Starting evaluation for individual {self.id} for {allocated_train_time} secs")
        evaluation_metrics: EvaluationMetrics = cnn_eval.evaluate(phenotype,
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
        logger.info(
            f"Evaluation results for individual {self.id}: {self.metrics}\n")

        assert self.fitness is not None
        return self.fitness
