import itertools
import time
from typing import List, Optional

from optimization.parameter import Parameter


class ConfigGenerator:
    """
    A generator providing configurations based on a list of parameter values.
    """

    def __init__(self, parameters: List[Parameter], seeds: Optional[List[int]] = None):
        """
        Init a new config generator with the given list of parameters and seeds (optional).

        :param parameters: the list of parameters
        :param seeds: the seeds or None
        """
        self.parameters = sorted(parameters, key=lambda p: p.name)
        self.seeds = seeds

    def get_parameter_names(self):
        """
        Get the name of each parameter.

        :return: the names
        """
        names = [parameter.name for parameter in self.parameters]
        names.insert(0, "seed")
        return names

    def __iter__(self):
        """
        Creates all combinations of configurations from the parameters and yields them. If no seeds were provided,
        the current UNIX time is used as seed instead.

        :return: the configurations
        """
        all_parameters = [list(parameter) for parameter in self.parameters]
        # Every configuration is run under every seed. The previous version
        # indexed self.seeds by the combination number, so a single-valued
        # parameter grid consumed only seeds[0] and silently discarded the rest,
        # and a grid with more combinations than seeds raised IndexError.
        seeds = self.seeds if self.seeds is not None else [None]
        for combination in itertools.product(*all_parameters):
            config = {
                parameter.name: combination[j]
                for j, parameter in enumerate(self.parameters)
            }
            for seed in seeds:
                seeded = dict(config)
                seeded["seed"] = int(time.time()) if seed is None else seed
                yield seeded
