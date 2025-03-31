import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess medical image for analysis.
    
    Args:
        image_path: Path to the medical image file
        
    Returns:
        numpy.ndarray: Processed image ready for model input
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image if too large (Gemini has size limits)
    max_dimension = 1024
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Apply slight Gaussian blur to reduce noise
    enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    
    # Ensure image is in correct format for Gemini
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    
    return enhanced_image

def draw_anomalies(image, detection_results):
    """
    Draw bounding boxes around detected anomalies.
    
    Args:
        image: Original image
        detection_results: Results from anomaly detection
        
    Returns:
        numpy.ndarray: Image with anomalies highlighted
    """
    result_image = image.copy()
    
    if detection_results.get("has_anomaly", False):
        for region in detection_results.get("regions", []):
            x, y = region["x"], region["y"]
            w, h = region["width"], region["height"]
            confidence = region.get("score", 0)
            
            # Draw rectangle
            cv2.rectangle(
                result_image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),  # Red color
                2
            )
            
            # Draw confidence text
            cv2.putText(
                result_image,
                f"{confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
    
    return result_image 