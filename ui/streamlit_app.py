import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json
from pathlib import Path

from models.classifier import ImageClassifier
from models.genai_helper import GeminiHelper
from utils.image_processing import preprocess_image, draw_anomalies
from utils.report_generator import generate_report
from config import GEMINI_API_KEY

def main():
    # Set page config
    st.set_page_config(
        page_title="Medical Vision Tool",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load translations
    def load_translations():
        translations = {
            "en": {
                "title": "Medical Vision Diagnostic Tool",
                "subtitle": "AI-Powered Medical Image Analysis",
                "upload_text": "Upload Medical Image",
                "patient_info": "Patient Information",
                "name": "Patient Name",
                "age": "Age",
                "gender": "Gender",
                "village": "Village",
                "district": "District",
                "state": "State",
                "analyze": "Analyze Image",
                "save_report": "Save Report",
                "loading": "Analyzing image...",
                "error": "Error",
                "success": "Success",
                "report_saved": "Report saved successfully!",
                "select_language": "Select Language",
                "gender_options": ["Male", "Female", "Other"],
                "analysis_results": "Analysis Results",
                "recommendations": "Recommendations",
                "medical_advice": "Medical Advice",
                "contact_doctor": "Please contact a doctor for detailed examination",
                "emergency_contact": "Emergency Contact",
                "phone": "Phone Number",
                "address": "Address",
                "save_patient": "Save Patient Information",
                "clear": "Clear Form",
                "upload_help": "Supported formats: JPG, JPEG, PNG, BMP",
                "patient_history": "Patient History",
                "no_history": "No previous records found",
                "new_analysis": "New Analysis",
                "previous_analyses": "Previous Analyses"
            },
            "hi": {
                "title": "चिकित्सा दृष्टि नैदानिक उपकरण",
                "subtitle": "एआई-संचालित चिकित्सा छवि विश्लेषण",
                "upload_text": "चिकित्सा छवि अपलोड करें",
                "patient_info": "रोगी की जानकारी",
                "name": "रोगी का नाम",
                "age": "आयु",
                "gender": "लिंग",
                "village": "गाँव",
                "district": "जिला",
                "state": "राज्य",
                "analyze": "छवि का विश्लेषण करें",
                "save_report": "रिपोर्ट सहेजें",
                "loading": "छवि का विश्लेषण कर रहा है...",
                "error": "त्रुटि",
                "success": "सफल",
                "report_saved": "रिपोर्ट सफलतापूर्वक सहेजी गई!",
                "select_language": "भाषा चुनें",
                "gender_options": ["पुरुष", "महिला", "अन्य"],
                "analysis_results": "विश्लेषण परिणाम",
                "recommendations": "सिफारिशें",
                "medical_advice": "चिकित्सा सलाह",
                "contact_doctor": "विस्तृत जांच के लिए कृपया डॉक्टर से संपर्क करें",
                "emergency_contact": "आपातकालीन संपर्क",
                "phone": "फोन नंबर",
                "address": "पता",
                "save_patient": "रोगी की जानकारी सहेजें",
                "clear": "फॉर्म साफ़ करें",
                "upload_help": "समर्थित प्रारूप: JPG, JPEG, PNG, BMP",
                "patient_history": "रोगी का इतिहास",
                "no_history": "कोई पिछला रिकॉर्ड नहीं मिला",
                "new_analysis": "नया विश्लेषण",
                "previous_analyses": "पिछले विश्लेषण"
            }
        }
        return translations

    # Initialize session state
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = {}
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

    # Load translations
    translations = load_translations()
    t = translations[st.session_state.language]

    # Sidebar
    with st.sidebar:
        st.title(t["select_language"])
        language = st.radio("", ["English", "हिंदी"], index=0 if st.session_state.language == 'en' else 1)
        st.session_state.language = 'en' if language == "English" else 'hi'
        t = translations[st.session_state.language]

    # Main content
    st.title(t["title"])
    st.subheader(t["subtitle"])

    # Create tabs
    tab1, tab2 = st.tabs([t["new_analysis"], t["previous_analyses"]])

    with tab1:
        # Patient Information Form
        st.header(t["patient_info"])
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input(t["name"], key="patient_name")
            age = st.number_input(t["age"], min_value=1, max_value=120, key="age")
            gender = st.selectbox(t["gender"], t["gender_options"], key="gender")
            village = st.text_input(t["village"], key="village")
        
        with col2:
            district = st.text_input(t["district"], key="district")
            state = st.text_input(t["state"], key="state")
            phone = st.text_input(t["phone"], key="phone")
            address = st.text_area(t["address"], key="address")
        
        # Emergency Contact
        st.subheader(t["emergency_contact"])
        emergency_phone = st.text_input(t["phone"], key="emergency_phone")
        
        # Image Upload
        st.header(t["upload_text"])
        st.info(t["upload_help"])
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button(t["analyze"], key="analyze_button"):
                with st.spinner(t["loading"]):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Initialize models
                        classifier = ImageClassifier()
                        genai_helper = GeminiHelper()
                        
                        # Process image
                        preprocessed = preprocess_image(temp_path)
                        cv_results = classifier.detect_anomalies(preprocessed)
                        genai_results = genai_helper.analyze_medical_image(temp_path, cv_results)
                        
                        # Generate report
                        report = generate_report(temp_path, cv_results, genai_results)
                        
                        # Store analysis in history
                        analysis_record = {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "patient_name": patient_name,
                            "report": report,
                            "image_path": temp_path
                        }
                        st.session_state.analysis_history.append(analysis_record)
                        
                        # Display results
                        st.success(t["success"])
                        st.markdown(report)
                        
                        # Save report button
                        if st.button(t["save_report"], key="save_report_button"):
                            # Create reports directory if it doesn't exist
                            reports_dir = Path("reports")
                            reports_dir.mkdir(exist_ok=True)
                            
                            # Save report
                            report_path = reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                            with open(report_path, "w", encoding="utf-8") as f:
                                f.write(report)
                            st.success(t["report_saved"])
                        
                    except Exception as e:
                        st.error(f"{t['error']}: {str(e)}")
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

    with tab2:
        # Display analysis history
        if not st.session_state.analysis_history:
            st.info(t["no_history"])
        else:
            for record in reversed(st.session_state.analysis_history):
                with st.expander(f"{record['date']} - {record['patient_name']}"):
                    st.markdown(record["report"])
                    if os.path.exists(record["image_path"]):
                        st.image(record["image_path"], caption="Analyzed Image", use_column_width=True)

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ for rural healthcare")

if __name__ == "__main__":
    main() 