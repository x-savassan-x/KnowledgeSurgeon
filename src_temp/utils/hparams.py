import yaml
from dataclasses import dataclass


@dataclass
class HyperParams:
    @classmethod
    def from_yaml(cls, fpath):
        with open(fpath, 'r') as f:
            config = yaml.safe_load(f)
            config = cls.construct_float_from_scientific_notation(config)
        return cls(**config)

    @classmethod
    def construct_float_from_scientific_notation(cls, config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    # Convert scalar to float if it is in scientific notation format
                    config[key] = float(value)
                except:
                    pass
        return config
