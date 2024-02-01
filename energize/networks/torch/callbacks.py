from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from time import time
from typing import TYPE_CHECKING, Any, Dict

import torch

from energize.misc.constants import (METADATA_FILENAME, MODEL_FILENAME,
                                     WEIGHTS_FILENAME)
from energize.misc.power import PowerConfig

if TYPE_CHECKING:
    from energize.networks.torch.trainers import Trainer


logger = logging.getLogger(__name__)


class Callback(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def on_train_begin(self, trainer: Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_train_end(self, trainer: Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_begin(self, trainer: Trainer) -> None:
        raise NotImplementedError()

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer) -> None:
        raise NotImplementedError()


# Needs tweak to save individual with lowest validation error
class ModelCheckpointCallback(Callback):
    def __init__(self,
                 model_saving_dir: str,
                 metadata_info: Dict[str, Any],
                 model_filename: str = MODEL_FILENAME,
                 weights_filename: str = WEIGHTS_FILENAME,
                 metadata_filename: str = METADATA_FILENAME) -> None:
        self.model_saving_dir: str = model_saving_dir
        self.model_filename: str = model_filename
        self.metadata_filename: str = metadata_filename
        self.weights_filename: str = weights_filename
        self.metadata_info: Dict[str, Any] = metadata_info

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        torch.save(trainer.model, os.path.join(
            self.model_saving_dir, self.model_filename))
        torch.save(trainer.model.state_dict(), os.path.join(
            self.model_saving_dir, self.weights_filename))
        with open(os.path.join(self.model_saving_dir, self.metadata_filename), 'w', encoding='utf-8') as f:
            json.dump(self._build_structured_metadata_json(self.metadata_info, trainer.trained_epochs),
                      f,
                      ensure_ascii=False,
                      indent=4)

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        pass

    def _build_structured_metadata_json(self,
                                        metadata_info: Dict[str, Any],
                                        trained_epochs: int) -> Dict[str, Any]:
        return {
            'dataset': {
                'name': metadata_info['dataset_name'],
            }
        }


class TimedStoppingCallback(Callback):
    def __init__(self, max_seconds: float) -> None:
        self.start_time: float = 0.0
        self.max_seconds: float = max_seconds

    def on_train_begin(self, trainer: Trainer) -> None:
        self.start_time = time()

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        if time() - self.start_time > self.max_seconds:
            trainer.stop_training = True


class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int) -> None:
        self.patience: int = patience
        self.best_score: float = 999999999.9
        self.counter: int = 0

    def on_train_begin(self, trainer: Trainer) -> None:
        self.counter = 0

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        if trainer.validation_loss[-1] >= self.best_score:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}. "
                        f"Best score {self.best_score}, current: {trainer.validation_loss[-1]}")
            if self.counter >= self.patience:
                trainer.stop_training = True
        else:
            self.best_score = trainer.validation_loss[-1]
            self.counter = 0


class PowerMeasureCallback(Callback):
    def __init__(self, power_config: PowerConfig) -> None:
        self.power_config: PowerConfig = power_config

    def on_train_begin(self, trainer: Trainer) -> None:
        self.power_config.meter.start(tag="train")

    def on_train_end(self, trainer: Trainer) -> None:
        self.power_config.meter.stop()

    def on_epoch_begin(self, trainer: Trainer) -> None:
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        pass
