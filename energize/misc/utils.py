import os
from argparse import ArgumentParser
from typing import Any, NewType

InputLayerId = NewType('InputLayerId', int)
LayerId = NewType('LayerId', int)

# pylint: disable=inconsistent-return-statements


def is_valid_file(parser: ArgumentParser, arg: Any) -> object:
    if not os.path.isfile(arg):
        parser.error(f"The file {arg} does not exist!")
    else:
        return arg

# pylint: disable=inconsistent-return-statements


def is_valid_config_file(parser: ArgumentParser, arg: Any) -> object:
    if is_valid_file(parser, arg) and arg.endswith((".yaml", ".json")):
        return arg
    parser.error(f"The file {arg} is not a yaml or json file")


class InvalidNetwork(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message
