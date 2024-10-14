import logging
import random
from collections import defaultdict, namedtuple
from copy import deepcopy
from itertools import chain
from operator import attrgetter
from turtle import distance
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from energize.evolution import Individual
from energize.evolution.grammar import (Derivation, Grammar, NonTerminal,
                                        Terminal)
from energize.misc import persistence
from energize.misc.enums import MutationType

if TYPE_CHECKING:
    from energize.evolution.grammar import Genotype, Symbol
    from energize.misc.fitness_metrics import Fitness
    from energize.networks.torch import BaseEvaluator

logger = logging.getLogger(__name__)


def mutation_dsge(layer: 'Genotype', grammar: Grammar, dynamic_bounds: Optional[Dict[str, tuple]] = None) -> None:
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
                symbol_to_mutate.attribute.update_bounds(
                    dynamic_bounds, symbol_to_mutate.name)
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
                    symbol.attribute.update_bounds(dynamic_bounds, symbol.name)
                    symbol.attribute.generate()
            layer.expansions[random_nt][nt_derivation_idx] = new_derivation
        else:
            raise AttributeError(
                f"Invalid value from random_mt_type: [{random_mt_type}]")


def mutation(individual: Individual,
             grammar: Grammar,
             generation: int,
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
    reuse_module: Optional[float] = mutation_config.get('reuse_module')
    remove_module: Optional[float] = mutation_config.get('remove_module')
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
        individual_copy.track_mutation(MutationType.TRAIN_LONGER, generation, {
                                       "from": individual_copy.total_allocated_train_time - default_train_time,
                                       "to": individual_copy.total_allocated_train_time
                                       })
        return individual_copy

    # in case the individual is mutated in any of the structural parameters
    # the training time is reset
    individual_copy.current_time = 0
    individual_copy.num_epochs = 0
    individual_copy.total_allocated_train_time = default_train_time
    individual_copy.metrics = None

    # reuse module
    if reuse_module is not None and random.random() <= reuse_module:
        individual_copy.reuse_module(grammar, generation)

    # remove module
    if remove_module is not None and random.random() <= remove_module:
        individual_copy.remove_module(grammar, generation)

    for m_idx, module in enumerate(individual_copy.modules):
        # add-layer (duplicate or new)
        for _ in range(random.randint(1, 2)):
            if random.random() <= add_layer_prob:
                module.add_layer(grammar, individual_copy, m_idx,
                                 reuse_layer_prob, generation)
        # remove-layer
        for _ in range(random.randint(1, 2)):
            if random.random() <= remove_layer_prob:
                module.remove_layer(
                    grammar, individual_copy, m_idx, generation)

        for layer_idx in range(len(module.layers)):
            # dsge mutation
            if random.random() <= dsge_layer_prob:
                module.layer_dsge(grammar, individual_copy,
                                  m_idx, layer_idx, generation)
            # add connection
            if layer_idx != 0 and random.random() <= add_connection_prob:
                module.layer_add_connection(grammar,
                                            individual_copy, m_idx, layer_idx, generation)
            # remove connection
            if layer_idx != 0 and random.random() <= remove_connection_prob:
                module.layer_remove_connection(grammar,
                                               individual_copy, m_idx, layer_idx, generation)

    dynamic_bounds = {
        'partition_point': (0, individual_copy.get_num_layers() - 1)
    }
    # macro level mutation
    for rule_idx, macro_rule in enumerate(individual_copy.macro_rules):
        requires_mutation, requires_mutation_reason = individual_copy.requires_dsge_mutation_macro(
            rule_idx, macro_rule)
        if random.random() <= macro_layer_prob or requires_mutation:
            old_macro_phenotype = grammar.decode(
                macro_rule, individual_copy.macro[rule_idx])
            mutation_dsge(
                individual_copy.macro[rule_idx], grammar, dynamic_bounds)
            track_mutation_data = {
                "from": old_macro_phenotype,
                "to": grammar.decode(macro_rule, individual_copy.macro[rule_idx])
            }
            if requires_mutation:
                track_mutation_data["observations"] = requires_mutation_reason
            individual_copy.track_mutation(
                MutationType.DSGE_MACRO, generation, track_mutation_data)
            logger.info(
                "Individual %d is going to have a macro mutation", individual_copy.id)
    return individual_copy


# Based on T. Helmuth, L. Spector and J. Matheson,
# "Solving Uncompromising Problems With Lexicase Selection,"
# in IEEE Transactions on Evolutionary Computation
def _lexicase_selection(population: List[Individual], metrics: List[dict]) -> Individual:
    candidates = [ind for ind in population if ind.metrics.is_valid_solution]
    cases = list(range(len(metrics)))
    random.shuffle(cases)

    while len(cases) > 0 and len(candidates) > 1:
        f = max if metrics[cases[0]]['objective'] == 'maximize' else min
        candidates_values = [ind.metrics.ordered_list([m['name'] for m in metrics])[
            cases[0]] for ind in candidates]
        best_val = f(candidates_values)
        candidates = [
            x for i, x in enumerate(candidates) if candidates_values[i] == best_val]
        logger.info("[Lexicase selection](metric: %s, objective: %s) -- %d candidates left.",
                    metrics[cases[0]]['name'], metrics[cases[0]]['objective'], len(candidates))
        cases.pop(0)

    return random.choice(candidates)

# Based on William La Cava, Lee Spector, and Kourosh Danai. 2016.
# Epsilon-Lexicase Selection for Regression. In Proceedings of the
# Genetic and Evolutionary Computation Conference 2016 (GECCO '16)


def _lexicase_auto_epsilon_selection(population: List[Individual], metrics: List[dict]) -> Individual:
    candidates = [ind for ind in population if ind.metrics.is_valid_solution]
    cases = list(range(len(metrics)))
    random.shuffle(cases)

    while len(cases) > 0 and len(candidates) > 1:
        candidates_values = [ind.metrics.ordered_list([m['name'] for m in metrics])[
            cases[0]] for ind in candidates]

        median_val = np.median(candidates_values, )
        median_abs_dev = np.median(
            [(abs(x - median_val)) for x in candidates_values])

        log: str
        if metrics[cases[0]]['objective'] == 'maximize':
            best_val = max(candidates_values)
            min_val_to_survive = best_val - median_abs_dev
            candidates = [x for i, x in enumerate(
                candidates) if candidates_values[i] >= min_val_to_survive]
            log = f"best:{best_val:.4f}, min_to_survive:{min_val_to_survive:.4f}"
        else:
            best_val = min(candidates_values)
            max_val_to_survive = best_val + median_abs_dev
            candidates = [x for i, x in enumerate(
                candidates) if candidates_values[i] <= max_val_to_survive]
            log = f"best:{best_val:.4f}, max_to_survive:{max_val_to_survive:.4f}"

        logger.info("[auto-ε-Lexicase selection](metric: %s, objective: %s, %s) -- %d candidates left.",
                    metrics[cases[0]]['name'], metrics[cases[0]]['objective'], log, len(candidates))
        cases.pop(0)

    return random.choice(candidates)


def fast_non_dominated_sort(individuals: List[Individual]) -> List[List[Individual]]:
    assert len(individuals) > 0

    n = [0] * len(individuals)
    rank = [0] * len(individuals)
    S = [[] for _ in individuals]
    front = [[]]

    for idx_p, p in enumerate(individuals):
        S[idx_p] = []
        n[idx_p] = 0
        for idx_q, q in enumerate(individuals):
            if p.metrics.dominates(q.metrics):
                if idx_q not in S[idx_p]:
                    S[idx_p].append(idx_q)
            elif q.metrics.dominates(p.metrics):
                n[idx_p] += 1
        if n[idx_p] == 0:
            rank[idx_p] = 0
            if idx_p not in front[0]:
                front[0].append(idx_p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)

    del front[-1]
    return front


def calculate_crowding_dist(individuals):
    assert len(individuals) > 0
    for ind in individuals:
        ind.crowding_dist = 0

    individuals[0].crowding_dist = float('inf')
    individuals[-1].crowding_dist = float('inf')

    sorted_accuracy = sorted(individuals, key=lambda x: x.metrics.accuracy)
    for i in range(1, len(individuals) - 2):
        individuals[i].crowding_dist += (sorted_accuracy[i + 1].metrics.accuracy -
                                         sorted_accuracy[i - 1].metrics.accuracy) / \
            (sorted_accuracy[-1].metrics.accuracy -
             sorted_accuracy[0].metrics.accuracy)

    sorted_power = sorted(
        individuals, key=lambda x: x.metrics.power["test"]["full"]["power"]["mean"])

    for i in range(1, len(individuals) - 2):
        individuals[i].crowding_dist += (sorted_power[i + 1].metrics.power["test"]["full"]["power"]["mean"] -
                                         sorted_power[i - 1].metrics.power["test"]["full"]["power"]["mean"]) / \
            (sorted_power[-1].metrics.power["test"]["full"]["power"]["mean"] -
             sorted_power[0].metrics.power["test"]["full"]["power"]["mean"])

def _nsga2_selection(population: List[Individual], k: int) -> Individual:
    """"
    [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.

    https://github.com/haris989/NSGA-II/blob/master/NSGA%20II.py
    """
    candidates = [ind for ind in population if ind.metrics.is_valid_solution]

    pareto_fronts = fast_non_dominated_sort(candidates)

    for front in pareto_fronts:
        calculate_crowding_dist(front)

    ...
    return


def select(population: List[Individual],
           method: str,
           method_params: Optional[dict],
           grammar: Grammar,
           cnn_eval: 'BaseEvaluator',
           run: int,
           generation: int,
           checkpoint_base_path: str,
           default_train_time: int) -> Individual:

    # Get best individual just according to fitness
    elite: Individual
    parent_10min: Individual

    parent: Individual
    idx_max: int
    if method == 'fittest':
        idx_max = np.argmax([ind.fitness for ind in population])
        parent = population[idx_max]
    elif method == 'lexicase':
        parent = _lexicase_selection(population, method_params['metrics'])
    elif method == 'lexicase-auto-epsilon':
        parent = _lexicase_auto_epsilon_selection(
            population, method_params['metrics'])
    elif method == "nsga2":
        parent = _nsga2_selection(population, 1)
    else:
        raise NotImplementedError(f"Method {method} not implemented")

    if method != "fittest":
        idx_max = population.index(parent)

    assert parent in population
    assert parent.fitness is not None
    logger.info("Parent: idx: %d, id: %d", idx_max, parent.id)
    logger.info("Training times: %s", str(
        [ind.current_time for ind in population]))
    logger.info("ids: %s", str([ind.id for ind in population]))

    # however if the parent is not the elite, and the parent is trained for longer, the elite
    # is granted the same evaluation time.
    if parent.total_allocated_train_time > default_train_time:
        retrain_elite = False
        if idx_max != 0 and population[0].total_allocated_train_time > default_train_time and \
                population[0].total_allocated_train_time < parent.total_allocated_train_time:
            logger.info(
                "Elite train was extended since parent was trained for longer")
            retrain_elite = True
            elite = population[0]
            assert elite.fitness is not None
            elite.total_allocated_train_time = parent.total_allocated_train_time
            elite.evaluate(grammar,
                           cnn_eval,
                           generation,
                           persistence.build_individual_path(
                               checkpoint_base_path, run, generation, elite.id),
                           persistence.build_individual_path(checkpoint_base_path, run, generation, elite.id))

        min_train_time = min(ind.current_time for ind in population)

        # also retrain the best individual that is trained just for the default time
        retrain_10min = False
        if min_train_time < parent.total_allocated_train_time:
            ids_10min = [ind.current_time ==
                         min_train_time for ind in population]
            logger.info(
                "Individuals trained for the minimum time: %s", str(ids_10min))
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
                parent_10min.evaluate(grammar,
                                      cnn_eval,
                                      generation,
                                      path,
                                      path)

        # select the fittest among all retrains and the initial parent
        if retrain_elite:
            assert elite.fitness is not None
            if retrain_10min:
                assert parent_10min.fitness is not None
                if parent_10min.fitness > elite.fitness and parent_10min.fitness > parent.fitness:
                    return deepcopy(parent_10min)
                if elite.fitness > parent_10min.fitness and elite.fitness > parent.fitness:
                    return deepcopy(elite)
                return deepcopy(parent)

            if elite.fitness > parent.fitness:
                return deepcopy(elite)

            return deepcopy(parent)
        if retrain_10min:
            assert parent_10min.fitness is not None
            if parent_10min.fitness > parent.fitness:
                return deepcopy(parent_10min)
            return deepcopy(parent)
        return deepcopy(parent)

    return deepcopy(parent)
