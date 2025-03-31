import google.generativeai as genai
import os
import base64
from PIL import Image
import io
from config import GEMINI_API_KEY

class GeminiHelper:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = GEMINI_API_KEY
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def encode_image(self, image):
        """Encode an image to base64 for Gemini API."""
        if isinstance(image, str):  # If image is a file path
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:  # If image is a numpy array
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                image = Image.fromarray(image)
            # Convert PIL Image to bytes
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def analyze_medical_image(self, image_path, detection_results=None):
        """
        Use Gemini to analyze a medical image and provide insights.
        
        Args:
            image_path: Path to the medical image or numpy array
            detection_results: Results from computer vision model
            
        Returns:
            dict: Analysis results from Gemini
        """
        try:
            # Prepare the prompt with medical context
            prompt = """
            You are a medical image analysis expert. Analyze this medical image and provide:
            1. Detailed description of visible medical features
            2. Potential abnormalities or concerns
            3. Possible diagnoses based on visual patterns
            4. Recommendations for further testing or specialist consultation
            5. Educational information about the potential conditions
            
            Be thorough but concise, and indicate confidence levels for your observations.
            Focus on medical accuracy and professional terminology.
            """
            
            # Add detection results to prompt if available
            if detection_results:
                prompt += f"\n\nComputer vision model detected anomalies with {detection_results['confidence']*100:.1f}% confidence."
            
            # Get a response from Gemini
            response = self.model.generate_content([
                prompt,
                {'mime_type': 'image/jpeg', 'data': self.encode_image(image_path)}
            ])
            
            # Check if response was blocked
            if response.prompt_feedback.block_reason:
                return {
                    "error": f"Content blocked: {response.prompt_feedback.block_reason}",
                    "analysis": "The image content was blocked by safety filters. Please ensure the image is appropriate for medical analysis.",
                    "confidence": 0
                }
            
            return {
                "analysis": response.text,
                "confidence": 0.85  # Placeholder confidence value
            }
            
        except Exception as e:
            error_message = str(e)
            if "API key" in error_message:
                return {
                    "error": "Invalid or missing API key",
                    "analysis": "Please check your Gemini API key configuration in config.py",
                    "confidence": 0
                }
            elif "image" in error_message.lower():
                return {
                    "error": "Image processing error",
                    "analysis": "The image could not be processed. Please ensure it's in a supported format (JPEG, PNG) and not corrupted.",
                    "confidence": 0
                }
            else:
                return {
                    "error": error_message,
                    "analysis": "An unexpected error occurred during analysis. Please try again.",
                    "confidence": 0
                } 