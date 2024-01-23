import unittest
import time

from energize.misc.power import PowerConfig, measure_power
from energize.networks.module import Module


class Test(unittest.TestCase):
    def setUp(self):
        self.num_measurements = 3
        self.power_config = PowerConfig(
            {"measure_power": {"num_measurements_test": self.num_measurements, "modules": True}})
        Module.power_config = self.power_config

    def test_power_measure(self):
        _, data_a = measure_power(self.power_config, time.sleep, (1,))
        _, data_b = measure_power(self.power_config, time.sleep, (2,))

        self.assertEqual(len(data_a["energy"]["data"]), self.num_measurements)
        self.assertNotEqual(data_a["energy"]["mean"], 0)
        self.assertNotEqual(data_b["energy"]["mean"], 0)
        self.assertGreater(data_b["energy"]["mean"], data_a["energy"]["mean"])


if __name__ == '__main__':
    unittest.main()
