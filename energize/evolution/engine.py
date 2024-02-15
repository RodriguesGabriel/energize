import logging
import random
from copy import deepcopy
from typing import List

import numpy as np
import torch

from energize.config import Config
from energize.evolution import Grammar, Individual, operators
from energize.misc import Checkpoint, persistence
from energize.misc.fitness_metrics import Fitness
from energize.networks.module import Module

logger = logging.getLogger(__name__)


@persistence.SaveCheckpoint
def evolve(run: int,
           grammar: Grammar,
           generation: int,
           checkpoint: Checkpoint,
           config: Config) -> Checkpoint:

    logger.info("Performing generation: %d", generation)
    population: List[Individual]
    population_fits: List[Fitness]
    if generation == 0:
        logger.info("Creating the initial population")

        # create initial population
        if config.get('energize') and config['energize'].get('seed_model'):
            # load seed model
            logger.info("Loading seed model")
            seed_individual = Individual(config['network']['architecture'], 0,
                                         config['evolutionary']['track_mutations'], run)\
                .load(grammar, config['energize']['seed_model']['phenotype'],
                      config['network']['architecture']['macro_structure'])
            population = [seed_individual]
            for _id_ in range(1, config['evolutionary']['lambda']):
                population.append(operators.mutation(deepcopy(seed_individual),
                                                     grammar,
                                                     generation,
                                                     config['evolutionary']['mutation'],
                                                     config['network']['learning']['default_train_time']))
                population[_id_].id = _id_
        else:
            population = [
                Individual(config['network']['architecture'], _id_,
                           config['evolutionary']['track_mutations'], run)
                .initialise(grammar, config['network']['architecture']['reuse_layer'])
                for _id_ in range(config['evolutionary']['lambda'])
            ]

        # set initial population variables and evaluate population
        population_fits = []
        for idx, ind in enumerate(population):
            ind.current_time = 0
            ind.num_epochs = 0
            ind.total_training_time_spent = 0.0
            ind.total_allocated_train_time = config['network']['learning']['default_train_time']
            ind.id = idx
            population_fits.append(
                ind.evaluate(grammar,
                             checkpoint.evaluator,
                             generation,
                             persistence.build_individual_path(config['checkpoints_path'], run, generation, idx))
            )

    else:
        assert checkpoint.parent is not None

        logger.info("Applying mutation operators")

        lambd: int = config['evolutionary']['lambda']
        # generate offspring (by mutation)
        offspring_before_mutation: List[Individual] = [
            deepcopy(checkpoint.parent) for _ in range(lambd)]
        for idx, offspring in enumerate(offspring_before_mutation):
            offspring.total_training_time_spent = 0.0
            offspring.id = idx + 1
        offspring: List[Individual] = \
            [operators.mutation(ind,
                                grammar,
                                generation,
                                config['evolutionary']['mutation'],
                                config['network']['learning']['default_train_time'])
             for ind in offspring_before_mutation]

        assert checkpoint.parent is not None
        population = [deepcopy(checkpoint.parent)] + offspring

        # set elite variables to re-evaluation
        population[0].current_time = 0
        population[0].num_epochs = 0
        population[0].id = 0
        population[0].metrics = None

        # evaluate population
        population_fits = []
        for idx, ind in enumerate(population):
            population_fits.append(
                ind.evaluate(
                    grammar,
                    checkpoint.evaluator,
                    generation,
                    persistence.build_individual_path(
                        config['checkpoints_path'], run, generation, idx),
                    persistence.build_individual_path(config['checkpoints_path'],
                                                      run,
                                                      generation-1,
                                                      checkpoint.parent.id),
                )
            )

    logger.info("Selecting the fittest individual")
    # select parent
    parent = operators.select_fittest(
        population,
        population_fits,
        grammar,
        checkpoint.evaluator,
        run,
        generation,
        config['checkpoints_path'],
        config['network']['learning']["default_train_time"])
    assert parent.fitness is not None

    logger.info(f"Fitnesses: {population_fits}")

    # update best individual
    best_individual_path: str = persistence.build_individual_path(config['checkpoints_path'],
                                                                  run,
                                                                  generation,
                                                                  parent.id)
    if checkpoint.best_fitness is None or parent.fitness > checkpoint.best_fitness:
        checkpoint.best_fitness = parent.fitness
        persistence.save_overall_best_individual(best_individual_path, parent)
    best_test_acc: float = checkpoint.evaluator.testing_performance(
        best_individual_path)

    logger.info("Generation best test accuracy: %f", best_test_acc)

    logger.info("Best fitness of generation %d: %f",
                generation, max(population_fits).value)
    logger.info("Best overall fitness: %f\n\n\n",
                checkpoint.best_fitness.value)

    return Checkpoint(
        run=run,
        random_state=random.getstate(),
        numpy_random_state=np.random.get_state(),
        torch_random_state=torch.get_rng_state(),
        last_processed_generation=generation,
        total_epochs=checkpoint.total_epochs +
        sum(ind.num_epochs for ind in population),
        best_fitness=checkpoint.best_fitness,
        evaluator=checkpoint.evaluator,
        population=population,
        parent=parent,
        best_gen_ind_test_accuracy=best_test_acc,
        modules_history=Module.history,
        statistics_format=checkpoint.statistics_format
    )
