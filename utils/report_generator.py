def generate_report(image_path, cv_results, genai_results):
    """
    Generate a comprehensive medical report combining CV and GenAI results.
    
    Args:
        image_path: Path to the analyzed image
        cv_results: Results from computer vision model
        genai_results: Results from Gemini analysis
        
    Returns:
        str: Formatted report
    """
    # Get current date
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Check for errors in GenAI results
    genai_error = genai_results.get("error", None)
    genai_analysis = genai_results.get("analysis", "No AI analysis available.")
    
    # Format the report
    report = """
    # Medical Image Analysis Report
    
    ## Image Information
    - File: {0}
    - Analysis Date: {1}
    
    ## Computer Vision Analysis
    - Anomaly Detected: {2}
    - Confidence: {3:.1f}%
    - Number of Regions: {4}
    
    ## AI Diagnostic Assistance
    {5}
    
    ## Recommendations
    - This is an automated analysis and should be reviewed by a healthcare professional
    - Store this report with the image for future reference
    - If anomalies were detected, prompt medical follow-up is recommended
    """.format(
        image_path.split('/')[-1],
        current_date,
        "Yes" if cv_results.get("has_anomaly", False) else "No",
        cv_results.get("confidence", 0) * 100,
        len(cv_results.get("regions", [])),
        genai_analysis
    )
    
    # Add error information if present
    if genai_error:
        report += f"\n\n## Error Information\n- Error Type: {genai_error}\n- Please ensure the image is appropriate for medical analysis and try again."
    
    return report 