import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    """
    Loads model weights from a directory of safetensors files.
    Skips loading weights that do not map to Python parameters.
    (e.g. CPU expert weights are skipped because they are not initialized in nn.ModuleDict)
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                target_param_name = weight_name
                shard_id = None
                
                for k, v in packed_modules_mapping.items():
                    if k in weight_name:
                        target_substring, shard_id = v
                        target_param_name = weight_name.replace(k, target_substring)
                        break

                try:
                    param = model.get_parameter(target_param_name)
                except AttributeError:
                    # Module doesn't exist, this happens for offloaded CPU experts
                    continue
                except KeyError:
                    # Parameter doesn't exist in the dict
                    continue

                if param is None:
                    continue

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is not None:
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                else:
                    weight_loader(param, f.get_tensor(weight_name))
