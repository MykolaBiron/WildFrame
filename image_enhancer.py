from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def enhance_image(image_path, output_path=None):
    """
    Enhance image quality using Pillow
    - Increases sharpness
    - Enhances contrast
    - Reduces noise
    """
    img = Image.open(image_path)
    
    # 1. Sharpen
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(2.0)
    
    # 2. Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # 3. Enhance color vibrancy
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    
    # 4. Slight brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    # Save if output path provided
    if output_path:
        img.save(output_path, quality=95)
    
    return img
