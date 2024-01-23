from copy import deepcopy
import logging
import random
from typing import Dict, List, TYPE_CHECKING, Tuple

import numpy as np

from energize.evolution import Individual
from energize.evolution.grammar import Derivation, Grammar, NonTerminal, Terminal
from energize.misc import persistence

if TYPE_CHECKING:
    from energize.networks.torch import BaseEvaluator
    from energize.misc.fitness_metrics import Fitness
    from energize.evolution.grammar import Genotype, Symbol

logger = logging.getLogger(__name__)


def mutation_dsge(layer: 'Genotype', grammar: Grammar) -> None:
    nt_keys: List[NonTerminal] = sorted(list(layer.expansions.keys()))
    random_nt: NonTerminal = random.choice(nt_keys)
    nt_derivation_idx: int = random.randint(
        0, len(layer.expansions[random_nt])-1)
    nt_derivation: Derivation = layer.expansions[random_nt][nt_derivation_idx]

    sge_possibilities: List[List[Symbol]] = []
    node_type_possibilites: List[type[Symbol]] = []
    if len(grammar.grammar[random_nt]) > 1:
        all_possibilities: List[Tuple[Symbol, ...]] = \
            [tuple(derivation) for derivation in grammar.grammar[random_nt]]
        # exclude current derivation to avoid neutral mutation
        sge_possibilities = [list(d) for d in set(
            all_possibilities) - set([tuple(nt_derivation)])]
        node_type_possibilites.append(NonTerminal)

    terminal_symbols_with_attributes: List[Symbol] = \
        list(filter(lambda x: isinstance(x, Terminal)
             and x.attribute is not None, nt_derivation))

    if terminal_symbols_with_attributes:
        node_type_possibilites.extend([Terminal, Terminal])

    if node_type_possibilites:
        random_mt_type: type[Symbol] = random.choice(node_type_possibilites)
        if random_mt_type is Terminal:
            symbol_to_mutate: Symbol = random.choice(
                terminal_symbols_with_attributes)
            assert isinstance(symbol_to_mutate, Terminal) and \
                symbol_to_mutate.attribute is not None and \
                symbol_to_mutate.attribute.values is not None
            is_neutral_mutation: bool = True
            while is_neutral_mutation:
                current_values = tuple(symbol_to_mutate.attribute.values)
                symbol_to_mutate.attribute.generate()
                new_values = tuple(symbol_to_mutate.attribute.values)
                if current_values != new_values:
                    is_neutral_mutation = False
        elif random_mt_type is NonTerminal:
            # assignment with side-effect.
            # layer variable will also be affected
            new_derivation: Derivation = deepcopy(
                Derivation(random.choice(sge_possibilities)))
            # this line is here because otherwise the index function
            # will not be able to find the derivation after we generate values
            layer.codons[random_nt][nt_derivation_idx] = grammar.grammar[random_nt].index(
                new_derivation)
            for symbol in new_derivation:
                if isinstance(symbol, Terminal) and symbol.attribute is not None:
                    assert symbol.attribute.values is None
                    symbol.attribute.generate()
            layer.expansions[random_nt][nt_derivation_idx] = new_derivation
        else:
            raise AttributeError(
                f"Invalid value from random_mt_type: [{random_mt_type}]")


def mutation(individual: Individual,
             grammar: Grammar,
             mutation_config: Dict[str, float],
             default_train_time: int) -> Individual:
    """
        Network mutations: add and remove layer, add and remove connections, macro structure


        Parameters
        ----------
        individual : Individual
            individual to be mutated

        grammar : Grammar
            Grammar instance, used to perform the initialisation and the genotype
            to phenotype mapping

        default_train_time : int
            default training time

        Returns
        -------
        ind : Individual
            mutated individual
    """

    add_layer_prob: float = mutation_config['add_layer']
    reuse_layer_prob: float = mutation_config['reuse_layer']
    remove_layer_prob: float = mutation_config['remove_layer']
    add_connection_prob: float = mutation_config['add_connection']
    remove_connection_prob: float = mutation_config['remove_connection']
    dsge_layer_prob: float = mutation_config['dsge_layer']
    macro_layer_prob: float = mutation_config['macro_layer']
    train_longer_prob: float = mutation_config['train_longer']
    individual_copy: Individual = deepcopy(individual)

    # Train individual for longer - no other mutation is applied
    if random.random() <= train_longer_prob:
        individual_copy.total_allocated_train_time += default_train_time
        logger.info("Individual %d total train time is going to be extended to %f",
                    individual_copy.id, individual_copy.total_allocated_train_time)
        return individual_copy

    # in case the individual is mutated in any of the structural parameters
    # the training time is reset
    individual_copy.current_time = 0
    individual_copy.num_epochs = 0
    individual_copy.total_allocated_train_time = default_train_time
    individual_copy.metrics = None

    for m_idx, module in enumerate(individual_copy.modules):
        # add-layer (duplicate or new)
        for _ in range(random.randint(1, 2)):
            if random.random() <= add_layer_prob:
                module.add_layer(grammar, individual_copy.id, m_idx,
                                 reuse_layer_prob)
        # remove-layer
        for _ in range(random.randint(1, 2)):
            if random.random() <= remove_layer_prob:
                module.remove_layer(grammar, individual_copy.id, m_idx)

        for layer_idx in range(len(module.layers)):
            # dsge mutation
            if random.random() <= dsge_layer_prob:
                module.layer_dsge(grammar, individual_copy.id,
                                  m_idx, layer_idx)
            # add connection
            if layer_idx != 0 and random.random() <= add_connection_prob:
                module.layer_add_connection(grammar,
                                            individual_copy.id, m_idx, layer_idx)
            # remove connection
            if layer_idx != 0 and random.random() <= remove_connection_prob:
                module.layer_remove_connection(grammar,
                                               individual_copy.id, m_idx, layer_idx)

    # macro level mutation
    for macro in individual_copy.macro:
        if random.random() <= macro_layer_prob:
            mutation_dsge(macro, grammar)
            logger.info(
                "Individual %d is going to have a macro mutation", individual_copy.id)

    return individual_copy


def select_fittest(population: List[Individual],
                   population_fits: List['Fitness'],
                   grammar: Grammar,
                   cnn_eval: 'BaseEvaluator',
                   run: int,
                   generation: int,
                   checkpoint_base_path: str,
                   default_train_time: int) -> Individual:  # pragma: no cover

    # Get best individual just according to fitness
    elite: Individual
    parent_10min: Individual
    idx_max: int = np.argmax(population_fits)  # type: ignore
    parent: Individual = population[idx_max]
    assert parent.fitness is not None
    logger.info(f"Parent: idx: {idx_max}, id: {parent.id}")
    logger.info(f"Training times: {[ind.current_time for ind in population]}")
    logger.info(f"ids: {[ind.id for ind in population]}")

    # however if the parent is not the elite, and the parent is trained for longer, the elite
    # is granted the same evaluation time.
    if parent.total_allocated_train_time > default_train_time:
        retrain_elite = False
        if idx_max != 0 and population[0].total_allocated_train_time > default_train_time and \
                population[0].total_allocated_train_time < parent.total_allocated_train_time:
            logger.info(
                "Elite train was extended, since parent was trained for longer")
            retrain_elite = True
            elite = population[0]
            assert elite.fitness is not None
            elite.total_allocated_train_time = parent.total_allocated_train_time
            elite.evaluate(grammar,
                           cnn_eval,
                           persistence.build_individual_path(
                               checkpoint_base_path, run, generation, elite.id),
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id))
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        # also retrain the best individual that is trained just for the default time
        retrain_10min = False
        if min_train_time < parent.total_allocated_train_time:
            ids_10min = [ind.current_time ==
                         min_train_time for ind in population]
            logger.info(
                f"Individuals trained for the minimum time: {ids_10min}")
            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                max_fitness_10min = max([ind.fitness for ind in indvs_10min])
                idx_max_10min = np.argmax([ind.fitness for ind in indvs_10min])
                parent_10min = indvs_10min[idx_max_10min]
                assert parent_10min.fitness is not None

                parent_10min.total_allocated_train_time = parent.total_allocated_train_time
                logger.info(f"Min train time parent: idx: {idx_max_10min}, id: {parent_10min.id},"
                            f" max fitness detected: {max_fitness_10min}")
                logger.info(f"Fitnesses from min train individuals before selecting best individual:"
                            f" {[ind.fitness for ind in indvs_10min]}")
                logger.info(f"Individual {parent_10min.id} has its train extended."
                            f" Current fitness {parent_10min.fitness}")
                path: str = persistence.build_individual_path(
                    checkpoint_base_path, run, generation, parent_10min.id)
                print(f"Reusing individual from path: {path}")
                print("Macro genotype from parent 10 min: ", parent_10min.macro)
                parent_10min.evaluate(grammar,
                                      cnn_eval,
                                      path,
                                      path)

                population_fits[population.index(
                    parent_10min)] = parent_10min.fitness

        # select the fittest among all retrains and the initial parent
        if retrain_elite:
            assert elite.fitness is not None
            if retrain_10min:
                assert parent_10min.fitness is not None
                if parent_10min.fitness > elite.fitness and parent_10min.fitness > parent.fitness:
                    return deepcopy(parent_10min)
                elif elite.fitness > parent_10min.fitness and elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
            else:
                if elite.fitness > parent.fitness:
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
        elif retrain_10min:
            assert parent_10min.fitness is not None
            if parent_10min.fitness > parent.fitness:
                return deepcopy(parent_10min)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)
