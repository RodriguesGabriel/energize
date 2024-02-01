from __future__ import annotations

import logging
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Subset
from torchvision import datasets

from energize.misc.proportions import ProportionsFloat, ProportionsIndexes
from energize.networks.torch.transformers import BaseTransformer

if TYPE_CHECKING:
    from torch.utils.data import Dataset

__all__ = ['DatasetType', 'load_dataset']

logger = logging.getLogger(__name__)


@unique
class DatasetType(Enum):
    EVO_TRAIN = "evo_train"
    EVO_VALIDATION = "evo_validation"
    EVO_TEST = "evo_test"
    TEST = "test"


def load_dataset(dataset_name: str,
                 train_transformer: BaseTransformer,
                 test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    dataset_mapping = {
        "fashion-mnist": _load_fashion_mnist,
        "mnist": _load_mnist,
        "cifar10": _load_cifar10,
        "cifar100": _load_cifar100
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return dataset_mapping[dataset_name](train_transformer, test_transformer)


def load_partitioned_dataset(seed: int,
                             dataset_name: str,
                             train_transformer: BaseTransformer,
                             test_transformer: BaseTransformer,
                             enable_stratify: bool,
                             proportions: Union[ProportionsFloat, ProportionsIndexes]) -> Dict[DatasetType, Subset]:

    train_data: Dataset
    evo_test_data: Dataset
    test_data: Dataset
    (train_data, evo_test_data, test_data) = load_dataset(
        dataset_name, train_transformer, test_transformer)

    subset_dict: Dict[DatasetType, Subset] = {}
    targets: Any
    targets_tensor: Tensor
    # In case we have received the indexes for each subset already
    if isinstance(proportions, ProportionsIndexes):
        for k in proportions.keys():
            if k == DatasetType.EVO_TEST:
                subset_dict[k] = Subset(evo_test_data, proportions[k])
            else:
                subset_dict[DatasetType.EVO_TRAIN] = Subset(
                    train_data, proportions[k])
        subset_dict[DatasetType.TEST] = Subset(test_data, list(
            range(len(test_data.targets))))  # type: ignore
        return subset_dict

    else:
        # otherwise we'll do it based on the proportions that were asked
        assert isinstance(proportions, ProportionsFloat)
        targets = train_data.targets  # type: ignore
        # pylint: disable=unidiomatic-typecheck
        targets_tensor = targets if type(
            targets) == "torch.Tensor" else Tensor(targets)
        train_indices: List[int] = list(range(len(targets)))  # * aug_factor

        stratify: Any = targets_tensor if enable_stratify else None
        train_idx, test_idx = train_test_split(train_indices,
                                               test_size=proportions[DatasetType.EVO_TEST],
                                               shuffle=True,
                                               stratify=stratify,
                                               random_state=seed)
        subset_dict[DatasetType.EVO_TEST] = Subset(evo_test_data, test_idx)

        if DatasetType.EVO_VALIDATION in proportions.keys():
            stratify = targets_tensor[train_idx] if enable_stratify else None
            real_validation_proportion: float = \
                proportions[DatasetType.EVO_VALIDATION] / \
                (1 - proportions[DatasetType.EVO_TEST])
            train_idx, valid_idx = train_test_split(train_idx,
                                                    test_size=real_validation_proportion,
                                                    shuffle=True,
                                                    stratify=stratify,
                                                    random_state=seed)
            subset_dict[DatasetType.EVO_VALIDATION] = Subset(
                train_data, valid_idx)

        subset_dict[DatasetType.EVO_TRAIN] = Subset(train_data, train_idx)
        subset_dict[DatasetType.TEST] = Subset(test_data, list(
            range(len(test_data.targets))))  # type: ignore

        return subset_dict


def _load_fashion_mnist(train_transformer: BaseTransformer,
                        test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, evo_test_data, test_data


def _load_mnist(train_transformer: BaseTransformer,
                test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )
    return train_data, evo_test_data, test_data


def _load_cifar10(train_transformer: BaseTransformer,
                  test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, evo_test_data, test_data


def _load_cifar100(train_transformer: BaseTransformer,
                   test_transformer: BaseTransformer) -> Tuple[Dataset, Dataset, Dataset]:
    train_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_transformer
    )
    evo_test_data = datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=test_transformer
    )
    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=test_transformer
    )

    return train_data, evo_test_data, test_data
