"""File for generating abcrown configs."""

from pathlib import Path
from typing import IO, Any
import tempfile
import yaml
from ConfigSpace import Configuration


def tmp_yaml_file() -> IO[str]:
    """Returns a new temporary named empty yaml file."""
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)

    return f


def tmp_yaml_file_from_dict(a_dict: dict[Any, Any]) -> IO[str]:
    """Returns a new temporary named yaml file with the dict written to it."""
    tmp_yaml = tmp_yaml_file()
    yaml.dump(a_dict, tmp_yaml)

    return tmp_yaml



def nested_set(dic: dict[Any, Any], keys: list[str], value: Any):
    """Set a nested dict value from a list of string keys."""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})

    dic[keys[-1]] = value



class AbcrownYamlConfig:
    """Class for ab-crown YAML configs."""

    def __init__(self, yaml_file: IO[str]):
        """New instance."""
        self._yaml_file = yaml_file

    @classmethod
    def from_yaml(
        cls,
        yaml_file: Path,
        network: Path,
        property: Path,
        timeout,
        *,
        batch_size: int = 64,
        yaml_override: dict[str, Any] | None = None,
    ):
        """Create new instance from a YAML file."""
        abcrown_dict = yaml.safe_load(yaml_file.read_text())

        nested_set(abcrown_dict, ["model", "onnx_path"], str(network))
        nested_set(
            abcrown_dict, ["specification", "vnnlib_path"], str(property)
        )
        nested_set(abcrown_dict, ["general", "save_adv_example"], True)
        nested_set(abcrown_dict, ["solver", "batch_size"], batch_size)

        if yaml_override:
            for k, v in yaml_override.items():
                nested_set(abcrown_dict, k.split("__"), v)

        new_yaml_file = tmp_yaml_file()
        yaml.dump(abcrown_dict, new_yaml_file)

        return cls(new_yaml_file)


    @classmethod
    def from_scratch(cls,
        network: Path,
        property: Path,
        timeout, 
        *,
        batch_size: int = 64,
        yaml_override: dict[str, Any] | None = None,):
        #TODO: not implemented
        yaml_file = tmp_yaml_file()

        #TODO: create from scratch 
        # attack:
#   attack_mode: PGD
#   enable_mip_attack: false
#   pgd_order: before
# bab:
#   branching:
#     input_split:
#       enable: false
#     method: kfsb
#     reduceop: min
# general:
#   complete_verifier: bab
#   enable_incomplete_verification: true
#   loss_reduction_func: sum
#   save_adv_example: true
# model:
#   onnx_path: /gpfs/home3/abosman/VERONA-fairness-project/VERONA/tests/test_experiment/data/networks/mnist-net_256x2.onnx
# solver:
#   batch_size: 512
#   bound_prop_method: alpha-crown
# specification:
#   vnnlib_path: /gpfs/home3/abosman/VERONA-fairness-project/VERONA/tests/test_experiment/ab_crown_one2any_14-01-2025+10_28/tmp/mnist-net_256x2/image_0/property_5_0_005.vnnlib


        return(cls(yaml_file))

    
    def get_yaml_file(self) -> IO[str]:
        """Get the ab-crown YAML config file."""
        if not self._yaml_file:
            raise FileNotFoundError("YAML file was not made yet.")

        return self._yaml_file

    def get_yaml_file_path(self) -> Path:
        """Get the path to the ab-crown YAML config file."""
        return Path(self.get_yaml_file().name)