# Medical Vision Diagnostic Tool

A portable diagnostic tool using computer vision and Gemini AI for medical image analysis.

## Features
- Image preprocessing using OpenCV
- Computer vision-based anomaly detection
- AI-powered analysis using Gemini
- User-friendly desktop interface
- Comprehensive report generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key:
   - Get an API key from Google AI Studio
   - Add it to `config.py` or set as environment variable `GEMINI_API_KEY`

3. Run the application:
```bash
python app.py
```

## Usage
1. Launch the application
2. Click "Load Image" to select a medical image
3. Click "Analyze" to process the image
4. View results and save report if needed

## Directory Structure
```
medical_vision_tool/
├── app.py                 # Main application
├── models/                # Model files
│   ├── __init__.py
│   ├── classifier.py      # Computer vision model
│   └── genai_helper.py    # Gemini integration 
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── image_processing.py
│   └── report_generator.py
├── ui/                    # User interface
│   ├── __init__.py
│   └── interface.py
├── data/                  # Sample data and cached images
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## Requirements
- Python 3.8+
- OpenCV
- TensorFlow
- Google Generative AI
- Other dependencies listed in requirements.txt 