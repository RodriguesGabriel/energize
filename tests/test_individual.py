# type: ignore
import random
import unittest
from typing import Any, Dict, List, Optional

from parameterized import parameterized

from energize.evolution.grammar import Genotype, Grammar
from energize.evolution.individual import Individual
from energize.networks import ModuleConfig
from tests.resources.genotype_examples import *  # pylint: disable=unused-wildcard-import,wildcard-import
from tests.resources.phenotype_examples import *  # pylint: disable=unused-wildcard-import,wildcard-import


class Test(unittest.TestCase):

    def _set_network_config(self, output: str) -> Dict[str, Any]:
        return {
            'reuse_layer': 0.2,
            'macro_structure': ['learning'],
            'output': output,
            'modules': {
                'features': ModuleConfig(min_expansions=3,
                                         max_expansions=30,
                                         initial_network_structure=[
                                             3, 4, 5, 6],
                                         levels_back=1)
            }
        }

    def setUp(self):
        grammar_path: str = "tests/resources/example_full.grammar"
        self.grammar: Grammar = Grammar(grammar_path)

    @parameterized.expand([
        (0, ind_test_output1, ind_test_macro1),
        (1, ind_test_output2, ind_test_macro2)
    ])
    def test_initialise(self, seed: int, expected_output: Genotype, expected_macro: Genotype):
        random.seed(seed)
        individual: Individual = Individual(
            network_architecture_config=self._set_network_config("softmax"),
            ind_id=seed,
            seed=seed
        )
        initialised_individual: Individual = individual.initialise(self.grammar,
                                                                   self._set_network_config("softmax")['reuse_layer'])

        self.assertEqual(initialised_individual.output, expected_output)
        self.assertEqual(initialised_individual.macro, [expected_macro])

    @parameterized.expand([
        (0, ind_phenotype1),
        (1, ind_phenotype2),
        # (1, ind_phenotype3),
        # (1, ind_phenotype4)
    ])
    def test_decode(self, seed: int, expected_phenotype: str):
        self.maxDiff = None
        random.seed(seed)
        network_config: Dict[str, Any]
        network_config = self._set_network_config("softmax")
        individual: Individual = Individual(
            network_architecture_config=network_config,
            ind_id=seed,
            seed=seed
        )
        initialised_individual: Individual = individual.initialise(self.grammar,
                                                                   network_config['reuse_layer'])
        # pylint: disable=protected-access
        phenotype: str = initialised_individual._decode(self.grammar)
        self.assertEqual(phenotype, expected_phenotype)


if __name__ == '__main__':
    unittest.main()
