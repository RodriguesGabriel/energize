from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from torch import Tensor

if TYPE_CHECKING:
    from energize.evolution import Individual
    from energize.misc.fitness_metrics import Fitness
    from energize.networks.module import Module
    from energize.networks.torch.evaluators import BaseEvaluator


@dataclass
class Checkpoint:
    run: int
    random_state: Any
    numpy_random_state: Dict[str, Any]
    torch_random_state: Tensor
    last_processed_generation: int
    total_epochs: int
    best_fitness: Optional[Fitness]
    evaluator: BaseEvaluator
    best_gen_ind_test_accuracy: float
    modules_history: List[Module]
    population: Optional[List[Individual]] = field(default=None)
    parent: Optional[Individual] = field(default=None)
    statistics_format: str = field(default=None)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Checkpoint):
            return self.__dict__ == other.__dict__
        return False
