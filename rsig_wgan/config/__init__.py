# %%

import yaml
from typing import Any
from pathlib import Path

def load_config() -> dict[str, Any]:
    """
    return: 
        Dictionary with the config
    """
    try:
        with open(Path(__file__).parent / "config.yml", 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: The file at config.yml was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None
    
__all__ = ["load_config"]
# %%
