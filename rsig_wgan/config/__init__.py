import yaml

def read_yaml_config() -> dict[str, Any]:
    """
    return: 
        Dictionary with the config
    """
    try:
        with open("config.yml", 'r') as file:
            config = yaml.safe_load(file)  # Parse the YAML file
            return config
    except FileNotFoundError:
        print(f"Error: The file at config.py was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None
    
__all__ = ["read_yaml_config"]