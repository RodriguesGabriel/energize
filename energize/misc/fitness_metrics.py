from __future__ import annotations

import math
import statistics as stats
from abc import ABC, abstractmethod
from sys import float_info
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader

from energize.misc.enums import Device, FitnessMetricName
from energize.misc.power import PowerConfig

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

__all__ = ["Fitness", "FitnessMetric", "AccuracyMetric",
           "LossMetric", "PowerMetric", "CustomFitnessFunction"]


class Fitness:
    def __init__(self, value: float, metric: type[FitnessMetric], power_data: Optional[dict] = None) -> None:
        self.value: float = value
        self.metric: type[FitnessMetric] = metric
        self.power: Optional[dict] = power_data

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Fitness):
            return self.__dict__ == other.__dict__
        return False

    def __lt__(self, other: Fitness) -> bool:
        return self.metric.worse_than(self, other)

    def __gt__(self, other: Fitness) -> bool:
        return self.metric.better_than(self, other)

    def __leq__(self, other: Fitness) -> bool:
        return self.metric.worse_or_equal_than(self, other)

    def __geq__(self, other: Fitness) -> bool:
        return self.metric.better_or_equal_than(self, other)

    def __str__(self) -> str:
        return str(round(self.value, 5))

    def __repr__(self) -> str:
        return self.__str__()


class FitnessMetric(ABC):
    def __init__(self, batch_size: Optional[int] = None, loss_function: Any = None) -> None:
        self.batch_size: Optional[int] = batch_size
        self.loss_function: Any = loss_function

    @abstractmethod
    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def worst_fitness(cls) -> Fitness:
        raise NotImplementedError()


class AccuracyMetric(FitnessMetric):
    def __init__(self, batch_size: Optional[int] = None, loss_function: Any = None) -> None:
        super().__init__(batch_size, loss_function)

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        model.eval()
        correct_guesses: int = 0
        size: int = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data[0].to(device.value, non_blocking=True), \
                    data[1].to(device.value, non_blocking=True)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs.data, 1)
                correct_guesses += (predicted ==
                                    labels).float().sum().item()
                size += len(labels)
        return correct_guesses / size

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1.0, cls)


class LossMetric(FitnessMetric):
    def __init__(self, loss_function: Any, batch_size: Optional[int] = None) -> None:
        super().__init__(batch_size, loss_function)

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> float:
        model.eval()
        total_loss: float
        n_batches: int = len(data_loader)
        with torch.no_grad():
            total_loss_tensor = torch.zeros(size=(1,), device=device.value)
            for _, data in enumerate(data_loader, 0):
                inputs, labels = data[0].to(device.value, non_blocking=True), \
                    data[1].to(device.value, non_blocking=True)
                outputs = model(inputs)
                total_loss += self.loss_function(outputs, labels)/n_batches
        total_loss = float(total_loss_tensor.data)
        if math.isinf(total_loss) or math.isnan(total_loss):
            raise ValueError(f"Invalid loss (inf or NaN): {total_loss}")
        else:
            return total_loss

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(float_info.max, cls)


class PowerMetric(FitnessMetric):
    def __init__(self, power_config: PowerConfig, return_in_joules: bool = False, batch_size: Optional[int] = None, loss_function: Any = None) -> None:
        super().__init__(batch_size, loss_function)
        self.return_in_joules: bool = return_in_joules
        self.power_config: PowerConfig = power_config
        self.power_data: Optional[dict] = None

    def compute_metric(self, model: nn.Module, data_loader: DataLoader, device: Device) -> dict:
        model.eval()

        n = self.power_config["measure_power"]["num_measurements_test"]
        measures = [0] * n
        durations = [0] * n

        for i in range(n):
            # start measuring power usage
            self.power_config.meter.start(tag="test")
            with torch.no_grad():
                for data in data_loader:
                    inputs, _ = data[0].to(device.value, non_blocking=True), \
                        data[1].to(device.value, non_blocking=True)
                    model(inputs)
            # stop measuring power usage
            self.power_config.meter.stop()
            # get power usage data
            trace = self.power_config.meter.get_trace()
            # convert power in mJ to J
            measures[i] = sum(trace[0].energy.values()) / 1000
            durations[i] = trace[0].duration
        power_data = np.divide(measures, durations)
        self.power_data = {
            "duration": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "data": durations
            },
            "energy": {
                "mean": np.mean(measures),
                "std": np.std(measures),
                "data": measures
            },
            "power": {
                "mean": np.mean(power_data),
                "std": np.std(power_data),
                "data": power_data.tolist()
            }
        }
        return self.power_data["energy"]["mean"] if self.return_in_joules else self.power_data["power"]["mean"]

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(float_info.max, cls)


class CustomFitnessFunction(FitnessMetric):
    def __init__(self, fitness_function: list[dict], power_config: PowerConfig, batch_size: Optional[int] = None, loss_function: Any = None) -> None:
        super().__init__(batch_size, loss_function)
        self.fitness_function = fitness_function
        self.power_config = power_config

    def compute_metric(self, model: Module, data_loader: DataLoader, device: Device) -> float:
        raise NotImplementedError()

    def compute_fitness(self, model: nn.Module, data_loader: DataLoader, device: Device, pre_computed: dict[str, float]) -> float:
        for param in self.fitness_function:
            if param["metric"] not in pre_computed:
                from energize.networks.torch.evaluators import \
                    create_fitness_metric

                metric_name, metric_data = FitnessMetricName.new(
                    param["metric"])
                metric = create_fitness_metric(
                    metric_name, metric_data, power_config=self.power_config)
                pre_computed[param["metric"]] = metric.compute_metric(
                    model, data_loader, device)

        return - pre_computed["power_0"] / pre_computed["accuracy_0"]

    @classmethod
    def worse_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value < other.value

    @classmethod
    def better_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value > other.value

    @classmethod
    def worse_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value <= other.value

    @classmethod
    def better_or_equal_than(cls, this: Fitness, other: Fitness) -> bool:
        return this.value >= other.value

    @classmethod
    def worst_fitness(cls) -> Fitness:
        return Fitness(-1, cls)
