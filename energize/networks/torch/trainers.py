import logging
import time
import traceback
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from energize.misc.enums import Device
from energize.misc.utils import InvalidNetwork
from energize.networks.torch.callbacks import Callback
from energize.networks.torch.lars import LARS


if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimiser: optim.Optimizer,
                 train_data_loader: DataLoader,
                 validation_data_loader: Optional[DataLoader],
                 loss_function: Any,
                 n_epochs: int,
                 initial_epoch: int,
                 device: Device,
                 callbacks: List[Callback] = [],
                 scheduler: Optional['LRScheduler'] = None) -> None:
        self.model: nn.Module = model
        self.optimiser: optim.Optimizer = optimiser
        self.loss_function: Any = loss_function
        self.train_data_loader: DataLoader = train_data_loader
        self.validation_data_loader: Optional[DataLoader] = validation_data_loader
        self.n_epochs: int = n_epochs
        self.initial_epoch: int = initial_epoch
        self.device: Device = device
        self.callbacks: List[Callback] = callbacks
        self.stop_training: bool = False
        self.trained_epochs: int = 0
        self.loss_values: Dict[str, List[float]] = {}
        self.validation_loss: List[float] = []
        self.scheduler: Optional['LRScheduler'] = scheduler

        # cuda stuff
        torch.cuda.empty_cache()
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = True

    def _call_on_train_begin_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_train_begin(self)

    def _call_on_train_end_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_train_end(self)

    def _call_on_epoch_begin_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def _call_on_epoch_end_callbacks(self) -> None:
        for c in self.callbacks:
            c.on_epoch_end(self)

    def train(self) -> None:
        # assert self.validation_data_loader is not None

        logging.info("Initiating supervised training")
        self.loss_values = {"train_loss": [], "val_loss": []}
        try:
            epoch: int = self.initial_epoch
            n_batches_train: int = len(self.train_data_loader)
            n_batches_validation: Optional[int]
            if self.validation_data_loader is not None:
                n_batches_validation = len(self.validation_data_loader)
            self.model.train()
            self._call_on_train_begin_callbacks()

            while epoch < self.n_epochs and self.stop_training is False:
                self._call_on_epoch_begin_callbacks()
                start = time.time()  # pylint: disable=unused-variable
                total_loss = torch.zeros(size=(1,), device=self.device.value)
                for i, data in enumerate(self.train_data_loader, 0):
                    inputs, labels = data[0].to(self.device.value, non_blocking=True), \
                        data[1].to(self.device.value, non_blocking=True)
                    if isinstance(self.optimiser, LARS):
                        self.optimiser.adjust_learning_rate(
                            n_batches_train, self.n_epochs, i)
                    # zero the parameter gradients
                    self.optimiser.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    total_loss += loss/n_batches_train
                    loss.backward()
                    self.optimiser.step()
                end = time.time()  # pylint: disable=unused-variable
                # logger.info(f"[{round(end-start, 2)}s] TRAIN epoch {epoch} -- loss: {total_loss}")
                self.loss_values["train_loss"].append(
                    round(float(total_loss.data), 3))

                if self.validation_data_loader is not None:
                    with torch.no_grad():
                        self.model.eval()
                        total_loss = torch.zeros(
                            size=(1,), device=self.device.value)
                        for i, data in enumerate(self.validation_data_loader, 0):
                            inputs, labels = data[0].to(self.device.value, non_blocking=True), \
                                data[1].to(self.device.value,
                                           non_blocking=True)
                            outputs = self.model(inputs)
                            total_loss += self.loss_function(
                                outputs, labels)/n_batches_validation
                        self.loss_values["val_loss"].append(
                            round(float(total_loss.data), 3))
                        # Used for early stopping criteria
                        self.validation_loss.append(float(total_loss.data))
                    self.model.train()
                    end = time.time()
                    # logger.info(f"[{round(end-start, 2)}s] VALIDATION epoch {epoch} -- loss: {total_loss}")
                if self.scheduler is not None:
                    self.scheduler.step()
                epoch += 1
                # logger.info("=============================================================")
                self._call_on_epoch_end_callbacks()

            self._call_on_train_end_callbacks()
            self.trained_epochs = epoch - self.initial_epoch
        except RuntimeError as e:
            print(e)
            print(traceback.format_exc())
            raise InvalidNetwork(str(e)) from e
