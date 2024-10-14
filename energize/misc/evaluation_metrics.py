from __future__ import annotations

from dataclasses import asdict, astuple, dataclass, fields
from typing import Any, Dict, Iterator, List, Optional

from energize.misc.enums import FitnessMetricName
from energize.misc.fitness_metrics import Fitness


@dataclass
class EvaluationMetrics:
    is_valid_solution: bool
    fitness: Fitness
    accuracy: Optional[tuple[float]]
    n_trainable_parameters: int
    n_layers: int
    training_time_spent: float
    losses: Dict[str, List[float]]
    n_epochs: int
    total_epochs_trained: int
    max_epochs_reached: bool
    power: Optional[dict]

    @classmethod
    def default(cls, fitness: Fitness) -> "EvaluationMetrics":
        return EvaluationMetrics(
            is_valid_solution=False,
            fitness=fitness,
            accuracy=None,
            n_trainable_parameters=-1,
            n_layers=-1,
            training_time_spent=0.0,
            losses={},
            n_epochs=0,
            total_epochs_trained=0,
            max_epochs_reached=False,
            power=None
        )

    def dominates(self, other: EvaluationMetrics):
        self_power = self.power["test"]["full"]["power"]["mean"]
        other_power = other.power["test"]["full"]["power"]["mean"]

        dominates_accuracy = self.accuracy >= other.accuracy
        dominates_power = self_power <= other_power

        return dominates_accuracy and dominates_power and (self.accuracy > other.accuracy or self_power < other_power)

    def list_fields(self) -> List[str]:
        return tuple(field.name for field in fields(self))

    def ordered_list(self, metrics) -> List[float]:
        l = []
        for m in metrics:
            m, value = FitnessMetricName.new(m)
            if m in (FitnessMetricName.ACCURACY, FitnessMetricName.ACCURACY_N):
                m = self.accuracy
                if value is not None:
                    m = m[value]
            elif m in (FitnessMetricName.POWER, FitnessMetricName.POWER_N):
                m = self.power["test"]
                if value is not None:
                    m = m[f"partition_{value}"]
                else:
                    m = m["full"]
                if m in (FitnessMetricName.ENERGY, FitnessMetricName.ENERGY_N):
                    m = m["energy"]["mean"]
                else:
                    m = m["power"]["mean"]
            else:
                raise ValueError(f"Unknown metric {m}")
            l.append(m)
        return l

    def __iter__(self) -> Iterator[Any]:
        data = list(astuple(self))
        data[1] = data[1].value

        return iter(data)

    def __str__(self) -> str:
        accuracy = None
        if self.accuracy is not None:
            if len(self.accuracy) > 1:
                accuracy = [round(a, 5) for a in self.accuracy]
            else:
                accuracy = round(self.accuracy[0], 5)

        return "EvaluationMetrics(" + \
            f"is_valid_solution: {self.is_valid_solution},  " + \
            f"n_trainable_parameters: {self.n_trainable_parameters},  " + \
            f"n_layers: {self.n_layers},  " + \
            f"training_time_spent: {self.training_time_spent},  " + \
            f"n_epochs: {self.n_epochs},  " + \
            f"total_epochs_trained: {self.total_epochs_trained},  " + \
            f"accuracy: {accuracy},  " + \
            f"fitness: {self.fitness},  " + \
            f"losses: {self.losses},  " + \
            f"power: {self.power},  " + \
            f"max_epochs_reached: {self.max_epochs_reached})"

    # To be used in case an individual gets extra training (through mutation or selection)
    def __add__(self, other: EvaluationMetrics) -> EvaluationMetrics:
        if self is None:
            return other

        max_epochs_reached: bool = self.max_epochs_reached or other.max_epochs_reached

        assert self.is_valid_solution == other.is_valid_solution
        assert self.n_trainable_parameters == other.n_trainable_parameters
        assert self.n_layers == other.n_layers

        # add should affect appending new losses,
        # adding the extra epochs trained and extra training time spent
        return EvaluationMetrics(
            is_valid_solution=self.is_valid_solution,
            fitness=other.fitness,
            accuracy=other.accuracy,
            n_trainable_parameters=self.n_trainable_parameters,
            n_layers=self.n_layers,
            training_time_spent=self.training_time_spent + other.training_time_spent,
            n_epochs=other.n_epochs,
            total_epochs_trained=self.total_epochs_trained + other.n_epochs,
            losses={k: self.losses[k] + other.losses[k]
                    for k in self.losses.keys()},
            max_epochs_reached=max_epochs_reached,
            power=other.power
        )
