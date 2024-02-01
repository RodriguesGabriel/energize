from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from typing import Any, Dict, Iterator, List, Optional

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

    def list_fields(self) -> List[str]:
        class_fields = [field.name for field in fields(self)]
        # if self.accuracy is not None and len(self.accuracy) > 1:
        #     idx = class_fields.index("accuracy")
        #     class_fields[idx] = "accuracy_0"
        #     for i in range(1, len(self.accuracy)):
        #         class_fields.insert(idx + i, f"accuracy_{i}")
        return class_fields

    def __iter__(self) -> Iterator[Any]:
        # data = list(astuple(self))
        # if self.accuracy is not None and len(self.accuracy) > 1:
        #     idx = data.index(self.accuracy)
        #     data[idx] = self.accuracy[0]
        #     for i in range(1, len(self.accuracy)):
        #         data.insert(idx + i, self.accuracy[i])
        # return iter(data)
        return iter(astuple(self))

    def __str__(self) -> str:
        return "EvaluationMetrics(" + \
            f"is_valid_solution: {self.is_valid_solution},  " + \
            f"n_trainable_parameters: {self.n_trainable_parameters},  " + \
            f"n_layers: {self.n_layers},  " + \
            f"training_time_spent: {self.training_time_spent},  " + \
            f"n_epochs: {self.n_epochs},  " + \
            f"total_epochs_trained: {self.total_epochs_trained},  " + \
            f"accuracy: {round(self.accuracy, 5) if self.accuracy is not None else None},  " + \
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
