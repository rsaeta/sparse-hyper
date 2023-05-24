import inspect
from dataclasses import dataclass

@dataclass
class _ConfigBase:

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })
