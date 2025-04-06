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
        
        # FIX: Store model properly
        self.model = genai.GenerativeModel(model_name='gemini-1.5-flash')  

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

    def analyze_medical_image(self, image_path, detection_results=None, language="en"):
        try:
            prompt = """
            You are a medical image analysis expert. Analyze this medical image and provide:
            1. Hospital Priority (RED/ORANGE/GREEN) and action (Immediate/Monitor/Home Care).
            2. Detailed description of visible features.
            3. Abnormalities or concerns.
            4. Possible diagnoses.
            5. Recommendations.
            """
            #Add language instruction
            if language == "hi":
                prompt += "\n\nRespond in Hindi language only."
            elif language == "ta":
                prompt += "\n\nRespond in Tamil language only."
            else:
                prompt += "\n\nRespond in English language."
            
            if detection_results:
                prompt += f"\n\nComputer vision model detected anomalies with {detection_results['confidence']*100:.1f}% confidence."

            # Now call Gemini
            response = self.model.generate_content([
                prompt,
                {'mime_type': 'image/jpeg', 'data': self.encode_image(image_path)}
            ])

            return {
                "analysis": response.text,
                "confidence": 0.85
            }

        except Exception as e:
            return {
                "error": str(e),
                "analysis": "An error occurred during medical image analysis.",
                "confidence": 0
            }
