from __future__ import annotations

import csv
import glob
import json
import os
import shutil
from typing import TYPE_CHECKING, Any, Callable, Optional

import dill

from energize.evolution import Individual
from energize.misc.constants import (MODEL_FILENAME, OVERALL_BEST_FOLDER,
                                     STATS_FOLDER_NAME)
from energize.misc.enums import Mutation
from energize.misc.evaluation_metrics import EvaluationMetrics
from energize.misc.power import PowerConfig
from energize.networks.module import Module

if TYPE_CHECKING:
    from energize.config import Config
    from energize.evolution.grammar import Grammar
    from energize.misc import Checkpoint


__all__ = ['RestoreCheckpoint', 'SaveCheckpoint', 'save_overall_best_individual',
           'build_individual_path', 'build_overall_best_path']



class RestoreCheckpoint:
    def __init__(self, f: Callable) -> None:
        self.f: Callable = f

    def __call__(self,
                 run: int,
                 dataset_name: str,
                 config: Config,
                 grammar: Grammar,
                 is_gpu_run: bool) -> None:
        self.f(run,
               dataset_name,
               config,
               grammar,
               is_gpu_run,
               possible_checkpoint=self.restore_checkpoint(config, run))

    def restore_checkpoint(self, config: Config, run: int) -> Optional[Checkpoint]:
        checkpoint_path = os.path.join(
            config['checkpoints_path'], f"run_{run}", "checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as handle_checkpoint:
                checkpoint: Checkpoint = dill.load(handle_checkpoint)
            Module.history = checkpoint.modules_history
            if config.get('energize', None) is not None:
                checkpoint.evaluator.power_config = PowerConfig(
                    config['energize'])
            return checkpoint
        return None


class SaveCheckpoint:
    def __init__(self, f: Callable) -> None:
        self.f: Callable = f

    def __call__(self, *args: Any, **kwargs: Any) -> Checkpoint:
        new_checkpoint: Checkpoint = self.f(*args)
        # we need to remove the power config from the checkpoint to save it
        temp = new_checkpoint.evaluator.power_config
        new_checkpoint.evaluator.power_config = None
        # we assume the config is the last parameter in the function decorated
        self._save_checkpoint(new_checkpoint,
                              args[-1]['checkpoints_path'],
                              args[-1]['evolutionary']['generations'])
        # restore the power config
        new_checkpoint.evaluator.power_config = temp
        return new_checkpoint

    def _save_checkpoint(self, checkpoint: Checkpoint, save_path: str, max_generations: int) -> None:
        assert checkpoint.population is not None
        assert checkpoint.parent is not None
        with open(os.path.join(save_path, f"run_{checkpoint.run}", "checkpoint.pkl"), "wb") as handle_checkpoint:
            dill.dump(checkpoint, handle_checkpoint)
        self._delete_unnecessary_files(checkpoint, save_path, max_generations)
        if checkpoint.statistics_format == "csv":
            self._save_statistics_csv(save_path, checkpoint)
        elif checkpoint.statistics_format == "json":
            self._save_statistics_json(save_path, checkpoint)
        else:
            raise ValueError(
                f"Unknown statistics format: {checkpoint.statistics_format}")

    # pylint: disable=unused-argument
    def _delete_unnecessary_files(self, checkpoint: Checkpoint, save_path: str, max_generations: int) -> None:
        assert checkpoint.population is not None
        # remove temporary files to free disk space
        files_to_delete = glob.glob(
            f"{save_path}/"
            f"run_{checkpoint.run}/"
            f"ind=*_generation={checkpoint.last_processed_generation}/*{MODEL_FILENAME}")
        for file in files_to_delete:
            os.remove(file)
        gen: int = checkpoint.last_processed_generation - 2
        if checkpoint.last_processed_generation > 1:
            folders_to_delete = glob.glob(
                f"{save_path}/run_{checkpoint.run}/ind=*_generation={gen}")
            for folder in folders_to_delete:
                shutil.rmtree(folder)
        # if checkpoint.last_processed_generation == max_generations-1:
        #    folders_to_delete = glob.glob(f"{save_path}/run_{checkpoint.run}/ind=*_generation=*")
        #    for folder in folders_to_delete:
        #        shutil.rmtree(folder)

    def _save_statistics_csv(self, save_path: str, checkpoint: Checkpoint) -> None:
        assert checkpoint.population is not None

        stats_path = os.path.join(save_path,
                                  f"run_{checkpoint.run}",
                                  STATS_FOLDER_NAME)

        with open(os.path.join(stats_path,
                               f"generation_{checkpoint.last_processed_generation}.csv"), 'w') as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(["id", "phenotype", "num_epochs", "total_training_time_allocated", "modules", "mutation_tracker"] +
                               checkpoint.population[0].metrics.list_fields())
            for ind in checkpoint.population:
                csvwriter.writerow([ind.id,
                                    ind.phenotype,
                                    ind.num_epochs,
                                    ind.total_allocated_train_time,
                                    ind.modules_phenotypes,
                                    ind.mutation_tracker,
                                    *ind.metrics])  # type: ignore

        test_accuracies_path = os.path.join(stats_path, "test_accuracies.csv")
        file_exists: bool = os.path.isfile(test_accuracies_path)
        with open(test_accuracies_path, 'a') as csvfile:
            csvwriter = csv.writer(
                csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if file_exists is False:
                csvwriter.writerow(["generation", "test_accuracy"])
            csvwriter.writerow(
                [checkpoint.last_processed_generation, checkpoint.best_gen_ind_test_accuracy])

    def _save_statistics_json(self, save_path: str, checkpoint: Checkpoint) -> None:
        assert checkpoint.population is not None

        json_obj = [{
            "id": ind.id,
            "phenotype": ind.phenotype,
            "num_epochs": ind.num_epochs,
            "total_training_time_allocated": ind.total_allocated_train_time,
            "modules": ind.modules_phenotypes,
            "mutation_tracker": ind.mutation_tracker,
            **dict(zip(ind.metrics.list_fields(), ind.metrics))
        } for ind in checkpoint.population]

        stats_path = os.path.join(save_path,
                                  f"run_{checkpoint.run}",
                                  STATS_FOLDER_NAME)

        with open(os.path.join(stats_path,
                               f"generation_{checkpoint.last_processed_generation}.json"), 'w') as file:
            json.dump(json_obj, file, indent=2, cls=Encoder)

        test_accuracies_path = os.path.join(stats_path,
                                            "test_accuracies.json")
        file_exists: bool = os.path.isfile(test_accuracies_path)

        if file_exists:
            with open(test_accuracies_path) as json_file:
                test_accuracies = json.load(json_file)
        else:
            test_accuracies = {}

        test_accuracies[checkpoint.last_processed_generation] = checkpoint.best_gen_ind_test_accuracy

        with open(test_accuracies_path, 'w') as file:
            json.dump(test_accuracies, file, indent=4)


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Mutation):
            return {
                'mutation_type': str(obj.mutation_type),
                'gen': obj.gen,
                'data': obj.data
            }
        return super().default(obj)

def save_overall_best_individual(best_individual_path: str, parent: Individual) -> None:
    # pylint: disable=unexpected-keyword-arg
    shutil.copytree(best_individual_path,
                    os.path.join(best_individual_path,
                                 "..", OVERALL_BEST_FOLDER),
                    dirs_exist_ok=True)
    with open(os.path.join(best_individual_path, "..", OVERALL_BEST_FOLDER, "parent.pkl"), "wb") as handle:
        dill.dump(parent, handle)


def build_individual_path(checkpoint_base_path: str,
                          run: int,
                          generation: int,
                          individual_id: int) -> str:
    return os.path.join(f"{checkpoint_base_path}",
                        f"run_{run}",
                        f"ind={individual_id}_generation={generation}")


def build_overall_best_path(checkpoint_base_path: str, run: int) -> str:
    return os.path.join(f"{checkpoint_base_path}", f"run_{run}", OVERALL_BEST_FOLDER)
