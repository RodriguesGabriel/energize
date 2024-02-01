import filecmp
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import validate

from energize.misc.enums import TransformOperation
from energize.networks import ModuleConfig

logger = logging.getLogger(__name__)


class Config():
    def __init__(self, path: str) -> None:
        self.config: Any = self._load(path)
        self._validate_config()
        self._fix_inconsistent_config()
        self.config['network']['architecture']['modules'] = self._convert_modules_configs_to_dict()
        os.makedirs(self.config['checkpoints_path'], exist_ok=True)
        self._backup_used_config(path, self.config['checkpoints_path'])

    def _convert_modules_configs_to_dict(self) -> Dict[str, ModuleConfig]:
        modules_configurations: Dict[str, ModuleConfig] = {}
        for module_info in self.config['network']['architecture']['modules']:
            modules_configurations[module_info['name']] = \
                ModuleConfig(min_expansions=module_info['network_structure'][0],
                             max_expansions=module_info['network_structure'][1],
                             initial_network_structure=module_info['network_structure_init'],
                             levels_back=module_info['levels_back'])
        return modules_configurations

    def _load(self, path: str) -> Any:
        with open(path, "r", encoding="utf8") as f:
            if path.endswith(".json"):
                return json.load(f)
            if path.endswith(".yaml"):
                return yaml.safe_load(f)
            raise ValueError(
                f"File extension not supported: {path.split('.')[-1]}")

    def _backup_used_config(self, origin_filepath: str, destination: str) -> None:
        extension = origin_filepath.split('.')[-1]
        destination_filepath: str = os.path.join(
            destination, f"used_config.{extension}")
        # if there is a config file backed up already and it is different than the one we are trying to backup
        if os.path.isfile(destination_filepath) and \
                filecmp.cmp(origin_filepath, destination_filepath) is False:
            raise ValueError("You are probably trying to continue an experiment "
                             "with a different config than the one you used initially. "
                             "This is a gentle reminder to double-check the config you "
                             "just passed as parameter.")
        # pylint: disable=protected-access
        if not shutil._samefile(origin_filepath, destination_filepath):  # type: ignore
            shutil.copyfile(origin_filepath, destination_filepath)

    def _validate_config(self) -> None:
        schema_path: str = os.path.join("energize", "config", "schema.json")
        schema: Any = self._load(schema_path)
        validate(self.config, schema)
        logger.info("Type of training: %s",
                    self.config['network']['learning']['learning_type'])
        if self.config['network']['learning']['learning_type'] == "supervised":
            if self.config['network']['learning']['augmentation']['train'] is not None:
                self._validate_augmentation_params(
                    self.config['network']['learning']['augmentation']['train'])
            logger.info("Augmentation used in training: %s",
                        self.config['network']['learning']['augmentation']['train'])
        else:
            raise ValueError(
                f"Learning type {self.config['network']['learning']['learning_type']} not supported")

        if self.config['network']['learning']['augmentation']['test'] is not None:
            self._validate_augmentation_params(
                self.config['network']['learning']['augmentation']['test'])

        logger.info("Augmentation used in test: %s",
                    self.config['network']['learning']['augmentation']['test'])

        if 'model_partition' in self.config['network']['architecture']['macro_structure'] \
                and 'energize' in self.config \
                and not self.config['energize']['model_partition']:
            raise ValueError(
                "'model_partition' was enabled in macro_structure but not enabled in the energize section ('model_partition' was set to False).")

    def _validate_augmentation_params(self, params: Dict[str, Any]) -> None:
        for key in params.keys():
            assert key in TransformOperation.enum_values(), \
                f"{key} is not recognised as one of the supported transforms"

    def _fix_inconsistent_config(self) -> None:
        if 'energize' in self.config:
            if self.config['energize']['model_partition'] \
                    and 'model_partition' not in self.config['network']['architecture']['macro_structure']:
                self.config['network']['architecture']['macro_structure'].append(
                    'model_partition')
                logger.warning(
                    "'model_partition' was not found in the macro_structure but it was active in the energize section. macro_structure was updated accordingly.")

    def __getitem__(self, key: str) -> Any:
        return self.config[key]
