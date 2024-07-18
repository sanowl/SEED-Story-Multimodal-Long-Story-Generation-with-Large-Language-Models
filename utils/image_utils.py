import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import requests
from io import BytesIO
from typing import Tuple, Union, Optional
import logging
from functools import lru_cache

@lru_cache(maxsize=100)
def load_and_preprocess_image(
    image_path: Union[str, bytes],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    data_format: str = 'channels_last',
    augment: bool = False,
    cache_key: Optional[str] = None
) -> np.ndarray:
    """
    Load an image from a file path, URL, or bytes, and preprocess it for model input.
    
    Args:
    image_path (str or bytes): Local file path, URL of the image, or image bytes.
    target_size (tuple): Target size for resizing the image.
    normalize (bool): Whether to normalize the image values to [-1, 1].
    data_format (str): 'channels_first' or 'channels_last'.
    augment (bool): Whether to apply data augmentation.
    cache_key (str, optional): Key for caching the result.
    
    Returns:
    np.ndarray: Preprocessed image as a numpy array.
    """
    try:
        img = _load_image(image_path)
        img = _preprocess_image(img, target_size, normalize, data_format, augment)
        return img
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise

def _load_image(image_path: Union[str, bytes]) -> Image.Image:
    if isinstance(image_path, str):
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            img = Image.open(image_path)
    elif isinstance(image_path, bytes):
        img = Image.open(BytesIO(image_path))
    else:
        raise ValueError("Invalid image_path type. Must be str or bytes.")
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def _preprocess_image(
    img: Image.Image,
    target_size: Tuple[int, int],
    normalize: bool,
    data_format: str,
    augment: bool
) -> np.ndarray:
    if augment:
        img = _apply_augmentation(img)
    
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if normalize:
        img_array = (img_array - 0.5) * 2
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    if data_format == 'channels_first':
        img_array = np.transpose(img_array, (2, 0, 1))
    elif data_format != 'channels_last':
        raise ValueError("Invalid data_format. Must be 'channels_first' or 'channels_last'.")
    
    return img_array

def _apply_augmentation(img: Image.Image) -> Image.Image:
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = ImageOps.mirror(img)
    
    # Random brightness adjustment
    brightness_factor = np.random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    
    # Random contrast adjustment
    contrast_factor = np.random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    
    return img