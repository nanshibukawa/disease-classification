import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier.logger import configure_logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

logger = configure_logger(__name__)
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox.
    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        BoxValueError: If the content of the YAML file cannot be converted to a ConfigBox.

    Returns:
        ConfigBox: Content of the YAML file wrapped in a ConfigBox. 
        If the file cannot be read, returns an empty ConfigBox.
    """
    if not path_to_yaml.exists():
        logger.error(f"YAML file not found at {path_to_yaml}")
        return ConfigBox({})
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file read successfully from {path_to_yaml}")
            if content is None:
                logger.warning(f"YAML file at {path_to_yaml} is empty or contains no valid data.")
                return ConfigBox({})
            # Convert the content to a ConfigBox for easier access
            if isinstance(content, dict):
                content = ConfigBox(content)
            elif isinstance(content, list):
                content = ConfigBox({"items": content})
            else:
                logger.warning(f"Unexpected content type in YAML file at {path_to_yaml}: {type(content)}")
                content = ConfigBox({"data": content})
            logger.info(f"YAML file content: {content}")
            return content
        
    except BoxValueError as e:
        logger.error(f"Error reading YAML file at {path_to_yaml}: {e}")
        return ConfigBox({})
    except Exception as e:
        logger.error(f"Unexpected error reading YAML file at {path_to_yaml}: {e}")
        return ConfigBox({})

@ensure_annotations
def create_directories(path_to_dir: list, verbose=True):
    """
    Creates directories specified in the list `path_to_dir`.
    Args:
        path_to_dir (list): List of directory paths to create.
        verbose (bool): If True, logs the creation of directories. Default is True.
    """
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Saves a dictionary to a JSON file.
    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to be saved in the JSON file.
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"JSON file saved at {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a dictionary from a JSON file.
    Args:
        path (Path): Path to the JSON file.
    
    Returns:
        dict: Data loaded from the JSON file.
    """
    with open(path, "r") as json_file:
        data = json.load(json_file)
    logger.info(f"JSON file loaded from {path}")
    return ConfigBox(data)

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Saves data to a binary file using joblib.
    Args:
        data (Any): Data to be saved.
        path (Path): Path to the binary file.
    """
    joblib.dump(data=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.
    Args:
        path (Path): Path to the binary file.
    
    Returns:
        Any: Data loaded from the binary file.
    """
    data = joblib.load(filename=path)
    logger.info(f"Binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """
Get size in KB
    Args:
        path (Path): Path to the file.
    
    Returns:
        str: Size of the file in KB, formatted to two decimal places.
    """
    size = round(os.path.getsize(path) / 1024, 2)
    return f"~ {size:.2f} KB"

def decodeImage(image_string: str, filename: str) -> None:
    """
    Decodes a base64 encoded image string and saves it to a file.
    
    Args:
        image_string (str): Base64 encoded image string.
        filename (str): Name of the file to save the decoded image.
    """
    imgdata = base64.b64decode(image_string)
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save the image data to the specified file
    with open(filename, "wb") as f:
        f.write(imgdata)
    logger.info(f"Image saved at {filename}")   

def encodeImageIntoBase64(croppedImagePath):

    """ Encodes an image file into a base64 string.     """
    if not os.path.exists(croppedImagePath):
        logger.error(f"Image file not found at {croppedImagePath}")
        return None
    with open(croppedImagePath, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
