from typing import Any, List
import statistics as stats
import numpy as np
import logging

import pynvml
from pyJoules.device import DeviceFactory, Device
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter

logger = logging.getLogger(__name__)

class PowerConfig:
    config: dict
    domains: List[NvidiaGPUDomain]
    devices: List[Device]
    meter: EnergyMeter
    measure_modules_power: bool

    def __init__(self, config: dict):
        self.config = config
        self.domains = [NvidiaGPUDomain(0)]
        self.devices = DeviceFactory.create_devices(self.domains)
        self.meter = EnergyMeter(self.devices)
        self.measure_modules_power = config["measure_power"]["modules"]

        if self.config["ensure_gpu_exclusivity"]:
            self._ensure_gpu_exclusivity()

    def _ensure_gpu_exclusivity(self):
        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            if pynvml.nvmlDeviceGetComputeMode(gpu) != pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
                logger.warning("GPU is not in exclusive compute mode. Attempting to set it now.")
                pynvml.nvmlDeviceSetComputeMode(gpu, pynvml.NVML_COMPUTEMODE_EXCLUSIVE_PROCESS)
                logger.warning("GPU is now in exclusive compute mode.")
            else:
                logger.info("GPU is in exclusive compute mode.")
        except pynvml.NVMLError as e:
            raise RuntimeError("You need root permissions to set the compute mode to exclusive.") from e
        finally:
            pynvml.nvmlShutdown()

    def __getitem__(self, key):
        return self.config[key]


def measure_power(power_config: PowerConfig, func, func_args):
    n = power_config["measure_power"]["num_measurements_test"]
    # average power usage of n runs of test step
    measures = [0] * n
    durations = [0] * n
    for i in range(n):
        # start measuring power usage
        power_config.meter.start(tag="test")
        # execute function
        output = func(*func_args)
        # stop measuring power usage
        power_config.meter.stop()
        # get power usage data
        trace = power_config.meter.get_trace()
        # convert power in mJ to J
        measures[i] = sum(trace[0].energy.values()) / 1000
        durations[i] = trace[0].duration
    return output, {
        "duration": {
            "mean": stats.mean(durations),
            "std": stats.stdev(durations),
            "data": durations
        },
        "energy": {
            "mean": stats.mean(measures),
            "std": stats.stdev(measures),
            "data": measures
        },
        "power": {
            "mean": stats.mean(np.divide(measures, durations)),
            "std": stats.stdev(np.divide(measures, durations)),
            "data": np.divide(measures, durations).tolist()
        }
    }


if __name__ == "__main__":
    import time
    print(measure_power(PowerConfig({}), time.sleep, (1,)))
