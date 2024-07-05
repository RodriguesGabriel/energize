from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Size, nn
from torch.utils.data import DataLoader, Subset

from energize.misc.constants import (DATASETS_INFO, MODEL_FILENAME,
                                     WEIGHTS_FILENAME)
from energize.misc.enums import Device, FitnessMetricName
from energize.misc.evaluation_metrics import *
from energize.misc.fitness_metrics import *
from energize.misc.phenotype_parser import Optimiser, parse_phenotype
from energize.misc.power import PowerConfig
from energize.misc.proportions import ProportionsFloat
from energize.misc.utils import InvalidNetwork
from energize.networks.torch.callbacks import (Callback, EarlyStoppingCallback,
                                               ModelCheckpointCallback,
                                               PowerMeasureCallback,
                                               TimedStoppingCallback)
from energize.networks.torch.dataset_loader import (DatasetType,
                                                    load_partitioned_dataset)
from energize.networks.torch.evolved_networks import EvolvedNetwork
from energize.networks.torch.trainers import Trainer
from energize.networks.torch.transformers import (BaseTransformer,
                                                  LegacyTransformer)

if TYPE_CHECKING:
    from energize.misc.phenotype_parser import ParsedNetwork
    from energize.networks.torch import LearningParams

__all__ = ['create_evaluator', 'BaseEvaluator', 'LegacyEvaluator']


logger = logging.getLogger(__name__)


def create_fitness_metric(metric_name: FitnessMetricName,
                          metric_data: Optional[int],
                          loss_function: Optional[Any] = None,
                          power_config: Optional[PowerConfig] = None) -> FitnessMetric:
    if metric_name is FitnessMetricName.ACCURACY \
            or FitnessMetricName.ACCURACY_N:
        return AccuracyMetric()
    if metric_name is FitnessMetricName.LOSS:
        assert loss_function is not None
        return LossMetric(loss_function)
    if metric_name is FitnessMetricName.POWER:
        return PowerMetric(power_config)
    if metric_name is FitnessMetricName.ENERGY:
        return PowerMetric(power_config, True)
    raise ValueError(f"Invalid fitness metric: [{metric_name}]")


def create_evaluator(dataset_name: str,
                     run: int,
                     evo_params: Dict[str, any],
                     learning_params: Dict[str, Any],
                     energize_params: Optional[Dict[str, Any]],
                     is_gpu_run: bool) -> 'BaseEvaluator':

    fitness_metric_name: Optional[FitnessMetricName] = None
    fitness_metric_data: Optional[int] = None
    fitness_function_params: Optional[list[dict]] = None
    selection: Optional[dict] = None

    if 'fitness_metric' in evo_params:
        fitness_metric_name, fitness_metric_data = FitnessMetricName.new(
            evo_params['fitness_metric'])
    elif 'fitness_function' in evo_params:
        fitness_function_params = evo_params['fitness_function']
    elif 'selection' in evo_params:
        selection = evo_params['selection']
        fitness_metric_name, fitness_metric_data = FitnessMetricName.new(
            'accuracy_0')

    train_transformer: Optional[BaseTransformer]
    test_transformer: Optional[BaseTransformer]

    user_chosen_device: Device = Device.GPU if is_gpu_run else Device.CPU
    learning_type: str = learning_params['learning_type']
    augmentation_params: Dict[str, Any] = learning_params['augmentation']
    data_splits_params: Dict[str, Any] = learning_params['data_splits']
    data_splits: Dict[DatasetType, float] = {
        DatasetType(k): v for k, v in data_splits_params.items()}

    if energize_params \
            and (energize_params['measure_power']['train'] or energize_params['measure_power']['test']):
        power_config = PowerConfig(energize_params)
        from energize.networks.module import Module
        Module.power_config = power_config
    else:
        power_config = None

    # Create Transformer instance
    if learning_type == 'supervised':
        augmentation_params['train'] = {
        } if augmentation_params['train'] is None else augmentation_params['train']
        train_transformer = LegacyTransformer(augmentation_params['train'])
        augmentation_params['test'] = {
        } if augmentation_params['test'] is None else augmentation_params['test']
        test_transformer = LegacyTransformer(augmentation_params['test'])
        return LegacyEvaluator(dataset_name,
                               fitness_metric_name,
                               fitness_metric_data,
                               fitness_function_params,
                               selection,
                               run,
                               user_chosen_device,
                               power_config,
                               train_transformer,
                               test_transformer,
                               data_splits)
    raise ValueError(f"Unexpected learning type: [{learning_type}]")


class BaseEvaluator(ABC):
    def __init__(self,
                 fitness_metric_name: Optional[FitnessMetricName],
                 fitness_metric_data: Optional[int],
                 fitness_function_params: Optional[list[dict]],
                 selection: Optional[dict],
                 seed: int,
                 user_chosen_device: Device,
                 dataset: Dict[DatasetType, Subset],
                 power_config: Optional[PowerConfig]) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        assert fitness_metric_name is not None or fitness_function_params is not None or selection is not None

        self.fitness_metric_name: Optional[FitnessMetricName] = fitness_metric_name
        self.fitness_metric_data: Optional[int] = fitness_metric_data
        self.fitness_function_params: Optional[list[dict]
                                               ] = fitness_function_params
        self.selection: Optional[dict] = selection
        self.seed: int = seed
        self.user_chosen_device: Device = user_chosen_device
        self.dataset = dataset
        self.power_config: Optional[PowerConfig] = power_config

    @staticmethod
    def _calculate_invalid_network_fitness(metric_name: Optional[FitnessMetricName]) -> Fitness:
        if metric_name is None:
            return CustomFitnessFunction.worst_fitness()
        if metric_name.value not in FitnessMetricName.enum_values():
            raise ValueError(
                f"Invalid fitness metric retrieved from the config: [{metric_name}]")
        if metric_name in (FitnessMetricName.ACCURACY, FitnessMetricName.ACCURACY_N):
            return AccuracyMetric.worst_fitness()
        if metric_name in (FitnessMetricName.POWER, FitnessMetricName.POWER_N):
            return PowerMetric.worst_fitness()
        if metric_name is FitnessMetricName.LOSS:
            return LossMetric.worst_fitness()
        raise ValueError("Invalid fitness metric")

    def _get_data_loaders(self,
                          dataset: Dict[DatasetType, Subset],
                          batch_size: int) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:

        g = torch.Generator()
        g.manual_seed(0)

        # during bt training if the the last batch has 1 element, training breaks at last batch norm.
        # therefore, we drop the last batch
        train_loader = DataLoader(dataset[DatasetType.EVO_TRAIN],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True,
                                  generator=g)

        validation_loader: Optional[DataLoader]
        if DatasetType.EVO_VALIDATION in dataset.keys():
            validation_loader = DataLoader(dataset[DatasetType.EVO_VALIDATION],
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           drop_last=False,
                                           pin_memory=True,
                                           generator=g)
        else:
            validation_loader = None

        test_loader = DataLoader(dataset[DatasetType.EVO_TEST],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 drop_last=False,
                                 pin_memory=True,
                                 generator=g)

        return train_loader, validation_loader, test_loader

    @staticmethod
    def decide_device(user_chosen_device: Device) -> Device:
        if user_chosen_device == Device.CPU:
            return Device.CPU
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            logger.warning(f"User chose training in {user_chosen_device.name} but CUDA/MPS is not available. "
                           f"Defaulting training to {Device.CPU.name}")
            return Device.CPU
        return Device.GPU

    @staticmethod
    def adapt_model_to_device(torch_model: nn.Module, device: Device) -> None:
        # if device == Device.GPU and torch.cuda.device_count() > 1:
        #     torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)
        torch.compile(torch_model, mode="reduce-overhead")

    @abstractmethod
    def evaluate(self,
                 phenotype: str,
                 first_individual_overall: bool,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics:
        raise NotImplementedError()

    def _build_callbacks(self,
                         model_saving_dir: str,
                         metadata_info: Dict[str, Any],
                         train_time: float,
                         early_stop: Optional[int],
                         power_config: Optional[PowerConfig]) -> List[Callback]:
        callbacks: List[Callback] = [ModelCheckpointCallback(model_saving_dir, metadata_info),
                                     TimedStoppingCallback(max_seconds=train_time)]
        if early_stop is not None:
            callbacks.append(EarlyStoppingCallback(patience=early_stop))
        if power_config is not None and power_config['measure_power']['train']:
            callbacks.append(PowerMeasureCallback(power_config))
        return callbacks

    def testing_performance(self, model_dir: str) -> float:
        model_filename: str
        weights_filename: str
        if isinstance(self, LegacyEvaluator):
            model_filename = MODEL_FILENAME
            weights_filename = WEIGHTS_FILENAME
        else:
            raise ValueError("Unexpected evaluator")

        torch_model: nn.Module = torch.load(
            os.path.join(model_dir, model_filename))
        torch_model.load_state_dict(torch.load(
            os.path.join(model_dir, weights_filename)))
        torch_model.eval()

        device: Device = self.decide_device(self.user_chosen_device)
        # if device == Device.GPU and torch.cuda.device_count() > 1:
        #     torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)

        test_set = self.dataset[DatasetType.TEST]
        assert test_set is not None
        test_loader: DataLoader = DataLoader(
            test_set, batch_size=64, shuffle=True)
        metric = AccuracyMetric(batch_size=64)
        return metric.compute_metric(torch_model, test_loader, device)


class LegacyEvaluator(BaseEvaluator):
    def __init__(self,
                 dataset_name: str,
                 fitness_metric_name: Optional[FitnessMetricName],
                 fitness_metric_data: Optional[int],
                 fitness_function_params: Optional[list[dict]],
                 selection: Optional[dict],
                 seed: int,
                 user_chosen_device: Device,
                 power_config: Optional[PowerConfig],
                 train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer,
                 data_splits: Dict[DatasetType, float]) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.dataset_name: str = dataset_name
        dataset: Dict[DatasetType, Subset] = load_partitioned_dataset(seed,
                                                                      dataset_name,
                                                                      train_transformer,
                                                                      test_transformer,
                                                                      enable_stratify=True,
                                                                      proportions=ProportionsFloat(
                                                                          data_splits))
        super().__init__(fitness_metric_name, fitness_metric_data, fitness_function_params, selection, seed,
                         user_chosen_device, dataset, power_config)

    def evaluate(self,
                 phenotype: str,
                 first_individual_overall: bool,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics:  # pragma: no cover

        # pylint: disable=cyclic-import,import-outside-toplevel
        from energize.networks.torch.model_builder import ModelBuilder

        optimiser: Optimiser
        device: Device = self.decide_device(self.user_chosen_device)
        torch_model: nn.Module
        fitness_value: Fitness
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        parsed_network, optimiser = parse_phenotype(phenotype)

        try:
            input_size: Tuple[int, int,
                              int] = DATASETS_INFO[self.dataset_name]["expected_input_dimensions"]
            model_builder: ModelBuilder = ModelBuilder(
                parsed_network, device, Size(list(input_size)))
            torch_model = model_builder.assemble_network(type(self))
            if parsed_network.data_type is not None:
                torch_model = torch_model.to(parsed_network.data_type)

            if reuse_parent_weights \
                    and parent_dir is not None \
                    and len(os.listdir(parent_dir)) > 0:
                torch_model.load_state_dict(torch.load(
                    os.path.join(parent_dir, WEIGHTS_FILENAME)))
            elif first_individual_overall \
                and self.power_config \
                    and self.power_config.get("seed_model") \
                    and self.power_config["seed_model"].get("weights_path"):
                logger.info("Loading seed model weights")
                torch_model.load_state_dict(torch.load(
                    self.power_config["seed_model"]["weights_path"]), strict=False)
            else:
                if reuse_parent_weights:
                    num_epochs = 0

            device = self.decide_device(self.user_chosen_device)
            self.adapt_model_to_device(torch_model, device)

            trainable_params_count: int = sum(
                p.numel() for p in torch_model.parameters() if p.requires_grad)
            if trainable_params_count == 0:
                raise InvalidNetwork(
                    "Network does not contain any trainable parameters.")

            learning_params: LearningParams = ModelBuilder.assemble_optimiser(
                torch_model.parameters(),
                optimiser
            )

            train_loader: DataLoader
            validation_loader: Optional[DataLoader]
            test_loader: DataLoader
            train_loader, validation_loader, test_loader = \
                self._get_data_loaders(
                    self.dataset, learning_params.batch_size)

            assert validation_loader is not None

            metadata_dict: Dict[str, Any] = {
                'dataset_name': self.dataset_name,
            }

            loss_function = nn.CrossEntropyLoss()
            trainer = Trainer(model=torch_model,
                              optimiser=learning_params.torch_optimiser,
                              loss_function=loss_function,
                              train_data_loader=train_loader,
                              validation_data_loader=validation_loader,
                              n_epochs=learning_params.epochs,
                              initial_epoch=num_epochs,
                              device=device,
                              callbacks=self._build_callbacks(model_saving_dir,
                                                              metadata_dict,
                                                              train_time,
                                                              learning_params.early_stop,
                                                              self.power_config))

            if train_time == 0:
                # to save the model
                trainer._call_on_train_begin_callbacks()
                trainer._call_on_train_end_callbacks()
            else:
                if self.power_config and self.power_config.get("model_partition"):
                    trainer.multi_output_train(
                        self.power_config["model_partition_n"])
                else:
                    trainer.train()

            power_data = {}
            # get training power measurements
            if self.power_config and self.power_config['measure_power']['train']:
                power_trace = self.power_config.meter.get_trace()[0]
                power_data['train'] = {
                    "duration": power_trace.duration,
                    "energy": sum(power_trace.energy.values()) / 1000,
                    "power": sum(power_trace.energy.values()) / 1000 / power_trace.duration,
                }

            model_partitions: Optional[List[EvolvedNetwork]] = None
            if self.power_config and self.power_config["model_partition"]:
                # get model partitions (the original model and the models with additional output)
                # the methods return a copy of the original model (but modified)
                model_partitions = [
                    torch_model.remove_additional_outputs(
                        model_builder.additional_output_idx),
                    *(torch_model.prune_unnecessary_layers(idx)
                      for idx in model_builder.additional_output_idx)
                ]

            fitness_metric: Optional[FitnessMetric |
                                     tuple[FitnessMetric]] = None
            fitness_metric_value: Optional[float | tuple[float]] = None
            if self.fitness_metric_name is not None:
                fitness_metric = create_fitness_metric(self.fitness_metric_name,
                                                       self.fitness_metric_data,
                                                       loss_function=loss_function,
                                                       power_config=self.power_config)
                if self.power_config and self.power_config["model_partition"]:
                    fitness_metric = (fitness_metric,) + tuple(
                        deepcopy(fitness_metric) for _ in model_partitions) if self.power_config["model_partition"] else fitness_metric
                    fitness_metric_value = tuple(
                        fm.compute_metric(mp, test_loader, device) for fm, mp in zip(fitness_metric, (torch_model, *model_partitions))
                    )
                else:
                    fitness_metric_value = fitness_metric.compute_metric(
                        torch_model, test_loader, device)

            if self.power_config and self.power_config['measure_power']['test']:
                if isinstance(fitness_metric, PowerMetric):
                    if self.power_config.get("model_partition"):
                        power_data['test'] = {
                            "full": fitness_metric[0].power_data
                        }
                        for i, fm in enumerate(fitness_metric[1:]):
                            power_data['test'][f"partition_{i}"] = fm.power_data
                    else:
                        power_data['test'] = fitness_metric.power_data
                else:
                    if self.power_config.get("model_partition"):
                        power_metric_full = PowerMetric(self.power_config)
                        power_metric_full.compute_metric(
                            torch_model, test_loader, device)

                        power_data['test'] = {
                            "full": power_metric_full.power_data
                        }

                        power_metric = tuple(
                            PowerMetric(self.power_config) for _ in model_partitions)
                        for i, (pm, mp) in enumerate(zip(power_metric, model_partitions)):
                            pm.compute_metric(mp, test_loader, device)
                            power_data['test'][f"partition_{i}"] = pm.power_data
                    else:
                        power_metric = PowerMetric(self.power_config)
                        power_metric.compute_metric(
                            torch_model, test_loader, device)
                        power_data['test'] = power_metric.power_data

            accuracy: Optional[float | tuple[float]]
            if fitness_metric is AccuracyMetric:
                if self.power_config and self.power_config.get("model_partition"):
                    accuracy = [None] * self.power_config["model_partition_n"]
                else:
                    accuracy = None
            else:
                if self.power_config and self.power_config["model_partition"]:
                    accuracy = tuple(AccuracyMetric().compute_metric(pm, test_loader, device)
                                     for pm in model_partitions)
                    accuracy = (np.mean(accuracy),) + accuracy
                else:
                    accuracy = AccuracyMetric().compute_metric(torch_model, test_loader, device)

            if self.fitness_function_params is not None:
                fitness_function = CustomFitnessFunction(
                    self.fitness_function_params, power_config=self.power_config)
                pre_computed = {}
                if accuracy is not None:
                    if self.power_config and self.power_config["model_partition"]:
                        pre_computed['accuracy'] = accuracy[0]
                        for i, acc in enumerate(accuracy[1:]):
                            pre_computed[f"accuracy_{i}"] = acc
                    else:
                        pre_computed['accuracy'] = accuracy
                if self.power_config and self.power_config['measure_power']['test']:
                    if self.power_config.get("model_partition"):
                        for i in range(self.power_config["model_partition_n"]):
                            pre_computed[f"energy_{i}"] = power_data['test'][f"partition_{i}"]["energy"]["mean"]
                            pre_computed[f"power_{i}"] = power_data['test'][f"partition_{i}"]["power"]["mean"]
                        pre_computed['energy'] = power_data['test']["full"]["energy"]["mean"]
                        pre_computed['power'] = power_data['test']["full"]["power"]["mean"]
                    else:
                        pre_computed['energy'] = power_data['test']["energy"]["mean"]
                        pre_computed['power'] = power_data['test']["power"]["mean"]
                fitness_value = Fitness(fitness_function.compute_fitness(
                    torch_model, test_loader, device, pre_computed), type(fitness_function))
            else:
                if self.selection is not None and isinstance(fitness_metric_value, tuple):
                    fitness_metric_value = fitness_metric_value[0]
                    fitness_metric = fitness_metric[0]
                fitness_value = Fitness(
                    fitness_metric_value, type(fitness_metric))

            n_layers: int | tuple[int] = (len(torch_model.evolved_layers),) + tuple(len(
                mp.evolved_layers) for mp in model_partitions) \
                if self.power_config and self.power_config.get("model_partition") \
                else len(parsed_network.layers)

            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                accuracy=accuracy,
                n_trainable_parameters=trainable_params_count,
                n_layers=n_layers,
                n_epochs=trainer.trained_epochs,
                losses=trainer.loss_values,
                training_time_spent=time()-start,
                total_epochs_trained=num_epochs+trainer.trained_epochs,
                max_epochs_reached=num_epochs+trainer.trained_epochs >= learning_params.epochs,
                power=power_data
            )
        except (InvalidNetwork, ValueError, IndexError):
            logger.warning(
                "Invalid model. Fitness will be computed as invalid individual.")
            fitness_value = self._calculate_invalid_network_fitness(
                self.fitness_metric_name)
            return EvaluationMetrics.default(fitness_value)
