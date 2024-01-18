from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch import nn, Size
from torch.utils.data import DataLoader, Subset

from energize.misc.constants import DATASETS_INFO, MODEL_FILENAME, WEIGHTS_FILENAME
from energize.misc.enums import Device, FitnessMetricName
from energize.misc.evaluation_metrics import EvaluationMetrics
from energize.misc.fitness_metrics import *  # pylint: disable=unused-wildcard-import,wildcard-import
from energize.misc.proportions import ProportionsFloat
from energize.misc.utils import InvalidNetwork
from energize.misc.phenotype_parser import parse_phenotype, Optimiser
from energize.misc.power import PowerConfig, measure_power
from energize.networks.torch.callbacks import Callback, EarlyStoppingCallback, \
    ModelCheckpointCallback, TimedStoppingCallback, PowerMeasureCallback
from energize.networks.torch.dataset_loader import DatasetType, load_partitioned_dataset
from energize.networks.torch.trainers import Trainer
from energize.networks.torch.transformers import BaseTransformer, LegacyTransformer

if TYPE_CHECKING:
    from energize.networks.torch import LearningParams
    from energize.misc.phenotype_parser import ParsedNetwork

__all__ = ['create_evaluator', 'BaseEvaluator', 'LegacyEvaluator']


logger = logging.getLogger(__name__)


def create_fitness_metric(metric_name: FitnessMetricName,
                          evaluator_type: type['BaseEvaluator'],
                          batch_size: Optional[int] = None,
                          loss_function: Optional[Any] = None) -> FitnessMetric:
    fitness_metric: FitnessMetric
    if metric_name.value not in FitnessMetricName.enum_values():
        raise ValueError(
            f"Invalid fitness metric retrieved from the config: [{metric_name}]")
    # print(evaluator_type, metric_name)
    if metric_name is FitnessMetricName.ACCURACY:
        fitness_metric = AccuracyMetric()
    elif metric_name is FitnessMetricName.LOSS:
        if evaluator_type is LegacyEvaluator:
            assert loss_function is not None
            fitness_metric = LossMetric(loss_function)
    else:
        raise ValueError(f"Unexpected evaluator type: [{evaluator_type}]")
    return fitness_metric


def create_evaluator(dataset_name: str,
                     fitness_metric_name: FitnessMetricName,
                     run: int,
                     learning_params: Dict[str, Any],
                     energize_params: Dict[str, Any],
                     is_gpu_run: bool) -> 'BaseEvaluator':

    train_transformer: Optional[BaseTransformer]
    test_transformer: Optional[BaseTransformer]

    user_chosen_device: Device = Device.GPU if is_gpu_run else Device.CPU
    learning_type: str = learning_params['learning_type']
    augmentation_params: Dict[str, Any] = learning_params['augmentation']
    data_splits_params: Dict[str, Any] = learning_params['data_splits']
    data_splits: Dict[DatasetType, float] = {
        DatasetType(k): v for k, v in data_splits_params.items()}

    if energize_params['measure_power']['train'] or energize_params['measure_power']['test']:
        power_config = PowerConfig(energize_params)
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
                               run,
                               user_chosen_device,
                               power_config,
                               train_transformer,
                               test_transformer,
                               data_splits)
    raise ValueError(f"Unexpected learning type: [{learning_type}]")


class BaseEvaluator(ABC):
    def __init__(self,
                 fitness_metric_name: FitnessMetricName,
                 seed: int,
                 user_chosen_device: Device,
                 dataset: Dict[DatasetType, Subset],
                 power_config: PowerConfig | None) -> None:
        """
            Creates the Evaluator instance and loads the dataset.

            Parameters
            ----------
            dataset : str
                dataset to be loaded
        """
        self.fitness_metric_name: FitnessMetricName = fitness_metric_name
        self.seed: int = seed
        self.user_chosen_device: Device = user_chosen_device
        self.dataset = dataset
        self.power_config: PowerConfig | None = power_config

    @staticmethod
    def _adapt_model_to_device(torch_model: nn.Module, device: Device) -> None:
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
        torch_model.to(device.value, non_blocking=True)
        torch.compile(torch_model, mode="reduce-overhead")

    @staticmethod
    def _calculate_invalid_network_fitness(metric_name: FitnessMetricName,
                                           evaluator_type: type['BaseEvaluator']) -> Fitness:
        if metric_name.value not in FitnessMetricName.enum_values():
            raise ValueError(
                f"Invalid fitness metric retrieved from the config: [{metric_name}]")
        if metric_name is FitnessMetricName.ACCURACY:
            return AccuracyMetric.worst_fitness()
        if metric_name is FitnessMetricName.LOSS:
            if evaluator_type is LegacyEvaluator:
                return LossMetric.worst_fitness()
            raise ValueError(f"Unexpected evaluator type: [{evaluator_type}]")
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

    def _decide_device(self) -> Device:
        if self.user_chosen_device == Device.CPU:
            return Device.CPU
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            logger.warning(f"User chose training in {self.user_chosen_device.name} but CUDA/MPS is not available. "
                           f"Defaulting training to {Device.CPU.name}")
            return Device.CPU
        return Device.GPU

    @abstractmethod
    def evaluate(self,
                 phenotype: str,
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
                         early_stop: int | None,
                         power_config: PowerConfig | None) -> List[Callback]:
        callbacks: List[Callback] = [ModelCheckpointCallback(model_saving_dir, metadata_info),
                                     TimedStoppingCallback(max_seconds=train_time)]
        if early_stop is not None:
            callbacks.append(EarlyStoppingCallback(patience=early_stop))
        if power_config is not None and power_config.config['measure_power']['train']:
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

        device: Device = self._decide_device()
        if device == Device.GPU and torch.cuda.device_count() > 1:
            torch_model = nn.DataParallel(torch_model)
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
                 fitness_metric_name: FitnessMetricName,
                 seed: int,
                 user_chosen_device: Device,
                 power_config: PowerConfig | None,
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
        super().__init__(fitness_metric_name, seed,
                         user_chosen_device, dataset, power_config)

    def evaluate(self,
                 phenotype: str,
                 model_saving_dir: str,
                 parent_dir: Optional[str],
                 reuse_parent_weights: bool,
                 train_time: float,
                 num_epochs: int) -> EvaluationMetrics:  # pragma: no cover

        # pylint: disable=cyclic-import,import-outside-toplevel
        from energize.networks.torch.model_builder import ModelBuilder

        optimiser: Optimiser
        device: Device = self._decide_device()
        torch_model: nn.Module
        fitness_value: Fitness
        start = time()

        os.makedirs(model_saving_dir, exist_ok=True)

        logger.info(phenotype)
        parsed_network, optimiser = parse_phenotype(phenotype)
        try:
            input_size: Tuple[int, int,
                              int] = DATASETS_INFO[self.dataset_name]["expected_input_dimensions"]
            model_builder: ModelBuilder = ModelBuilder(
                parsed_network, device, Size(list(input_size)))
            torch_model = model_builder.assemble_network(type(self))
            if reuse_parent_weights \
                    and parent_dir is not None \
                    and len(os.listdir(parent_dir)) > 0:
                torch_model.load_state_dict(torch.load(
                    os.path.join(parent_dir, WEIGHTS_FILENAME)))
            else:
                if reuse_parent_weights:
                    num_epochs = 0

            device = self._decide_device()
            self._adapt_model_to_device(torch_model, device)

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
            trainer.train()

            # get training power measurements
            if self.power_config.config['measure_power']['train']:
                power_trace = self.power_config.meter.get_trace()[0]
            else:
                power_trace = None

            fitness_metric: FitnessMetric = create_fitness_metric(self.fitness_metric_name,
                                                                  type(self),
                                                                  loss_function=loss_function)

            if self.power_config.config['measure_power']['test']:
                fitness_metric_value, power_data_test = measure_power(
                    self.power_config, fitness_metric.compute_metric, (torch_model, test_loader, device))
            else:
                fitness_metric_value = fitness_metric.compute_metric(
                    torch_model, test_loader, device)

            fitness_value = Fitness(fitness_metric_value, type(fitness_metric))
            accuracy: Optional[float]
            if fitness_metric is AccuracyMetric:
                accuracy = None
            else:
                accuracy = AccuracyMetric().compute_metric(torch_model, test_loader, device)

            power_data = {}
            if self.power_config.config['measure_power']['train']:
                power_data['train'] = {
                    "duration": power_trace.duration,
                    "energy": sum(power_trace.energy.values())
                }
            if self.power_config.config['measure_power']['test']:
                power_data['test'] = power_data_test

            return EvaluationMetrics(
                is_valid_solution=True,
                fitness=fitness_value,
                accuracy=accuracy,
                n_trainable_parameters=trainable_params_count,
                n_layers=len(parsed_network.layers),
                n_epochs=trainer.trained_epochs,
                losses=trainer.loss_values,
                training_time_spent=time()-start,
                total_epochs_trained=num_epochs+trainer.trained_epochs,
                max_epochs_reached=num_epochs+trainer.trained_epochs >= learning_params.epochs,
                power=power_data
            )
        except InvalidNetwork as e:
            logger.warning(
                "Invalid model. Fitness will be computed as invalid individual. Reason: %s", e.message)
            fitness_value = self._calculate_invalid_network_fitness(
                self.fitness_metric_name, type(self))
            return EvaluationMetrics.default(fitness_value)
